import os
import time
from time import sleep
from datetime import datetime, timedelta
import warnings
import random
from lorem_text import lorem
import threading
from threading import Lock
import webrtcvad
import numpy as np
import whisper
from whisper import Whisper
from faster_whisper import WhisperModel
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from queue import Queue
from engineio.payload import Payload
from tempfile import NamedTemporaryFile
import logging
import io

logging.getLogger("werkzeug").disabled = True

def print_message(msg: str):
    print(msg)
    logging.info(msg)

Payload.max_decode_packets = 500
UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
warnings.filterwarnings("ignore")

async_mode = 'eventlet'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

processor_thread = None
processor_thread_lock = Lock()

vad = webrtcvad.Vad()
vad.set_mode(3)
frames = b''

SAMPLE_RATE = 16000                     # hertz

# check if speech exists
MIN_CHUNK_SIZE = 480                    # in bytes
STEP_SIZE = 960                         # in bytes

# phrase complete and max buffer size (max_length=30s)
PHRASE_TIMEOUT = 5                      # in seconds
THIRTY_SECS_SIZE = 30*2*SAMPLE_RATE     # in bytes

CLOSE_REQUEST = "close"

# global variables
backend_started = False                 # boolean
input_queue = Queue()
combined_bytes = bytes()                # in bytes
last_sample_timestamp = None            # timestamp
cache_sample = None                     # {'timestamp':...,'data':...}
isMove2NextChunk = False                # each chunk maximum 30s
isPhraseComplete = False                # boolean
isFreshBytesAdded = False               # boolean
lastTranscribedText = None              # string
isFinal = False                         # session finished fired from client
sampling_count = 0                      # int
input_queue_count = 0                   # int

# get support language set for output
LANGUAGES = os.getenv('LANGUAGES') if os.getenv('LANGUAGES') else "vi,en"
LANGUAGES  = [item.strip() for item in LANGUAGES.split(',')]

# Get environment variables
WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL_NAME') if os.getenv('WHISPER_MODEL_NAME') else "tiny"
DEVICE = os.getenv("DEVICE") if os.getenv("DEVICE") else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") if os.getenv("COMPUTE_TYPE") else "auto"

print_message(f"MODEL_NAME={WHISPER_MODEL_NAME}")
print_message(f"DEVICE={DEVICE}")
print_message(f"COMPUTE_TYPE={COMPUTE_TYPE}")

# load whisper model
# model = whisper.load_model(WHISPER_MODEL_NAME)
model = WhisperModel(WHISPER_MODEL_NAME, 
                     device=DEVICE,
                     compute_type=COMPUTE_TYPE,
                     download_root="models/"
        )
print(f"Model whisper `{WHISPER_MODEL_NAME}` loaded.")


# if async_mode == 'eventlet':
#     import eventlet
#     eventlet.monkey_patch()

def buildTranscribedDataResponse():
    global lastTranscribedText
    global isMove2NextChunk
    global isPhraseComplete
    global isFinal

    return {
        'data': lastTranscribedText, 
        'isMove2NextChunk': isMove2NextChunk,
        'isPhraseComplete': isPhraseComplete, 
        'isFinal': isFinal
    }

def popAtMost30SecondLengthSample() -> bytearray:
    '''
    Pop as much items from queue (in bytes) but less than 30 secs or phrase complete timeout.
    - Input queue is required.
    '''
    global combined_bytes
    global last_sample_timestamp
    global cache_sample
    global isMove2NextChunk
    global isPhraseComplete
    global isFreshBytesAdded
    global isFinal
    global CLOSE_REQUEST
    global sampling_count

    # =========== SAMPLING =============
    isFreshBytesAdded = False
    sampling_count +=1
    print(F"==========> SAMPLING #{sampling_count}:")
    
    if isPhraseComplete:
        combined_bytes = bytes()
        last_sample_timestamp = None
        isPhraseComplete = False
    
    if isMove2NextChunk:
        combined_bytes = bytes()
        last_sample_timestamp = None
        isMove2NextChunk = False    

    if cache_sample:
        last_sample_timestamp = cache_sample['timestamp']
        combined_bytes = cache_sample['data']
        isFreshBytesAdded = True
        cache_sample = None
        print(f"<-...cache sample (len={len(combined_bytes)}) added!")

    if not last_sample_timestamp:
        first_sample = input_queue.get()
        
        # check if close event fired
        if first_sample['data'] == CLOSE_REQUEST:
            isFinal = True
            if not isFreshBytesAdded:
                combined_bytes = bytes()            
            
            print(f"-> Close request fired. Return with current bytes (len={len(combined_bytes)}).")
            print("<=====================================")
            return combined_bytes

        last_sample_timestamp = first_sample['timestamp']
        combined_bytes = first_sample['data']
        isFreshBytesAdded = True
        print(f"-> 1st sample (len={len(first_sample['data'])}) for a new chunk added!")

    while (not input_queue.empty()) and (len(combined_bytes) < THIRTY_SECS_SIZE):
        current_sample = input_queue.get()
        current_sample_timestamp = current_sample['timestamp']
        current_sample_data = current_sample['data']

        # check if close event fired
        if current_sample_data == CLOSE_REQUEST:
            isFinal = True
            if not isFreshBytesAdded:
                combined_bytes = bytes()
            
            print(f"-> Close request fired. Return with current bytes (len={len(combined_bytes)}).")
            print("<=====================================")
            return combined_bytes
        
        time_gap = datetime.utcfromtimestamp(current_sample_timestamp) \
                    - datetime.utcfromtimestamp(last_sample_timestamp)
        print(f"-> time_gap={time_gap}")
        
        if time_gap < timedelta(seconds=PHRASE_TIMEOUT):
            if len(combined_bytes) + len(current_sample_data) <= THIRTY_SECS_SIZE:
                
                print("-> +++appending to combined_bytes....")
                print(f"-> current bytes: {len(combined_bytes)}")
                print(f"-> adding bytes: {len(current_sample_data)}")
                print(f"-> resulted bytes: {len(combined_bytes) + len(current_sample_data)}")

                last_sample_timestamp = current_sample_timestamp
                combined_bytes += current_sample_data
                isFreshBytesAdded = True
            else:
                cache_sample = current_sample
                last_sample_timestamp = None
                isMove2NextChunk = True
                print("->...caching, move to next chunk....")
                break
        else:
            cache_sample = current_sample
            last_sample_timestamp = None
            isPhraseComplete = True
            print("->...caching, finish a phrase, new line....")
            break

    print(f"-> Sampling finalizing...")
    if not isFreshBytesAdded:
        print ("-> No fresh data added!")
    print(f"-> isMove2NextChunk: {isMove2NextChunk}")
    print(f"-> isPhraseComplete: {isPhraseComplete}")
    print(f"-> isFinal: {isFinal}")
    print(f"-> OUTPUT LENGTH: {len(combined_bytes)}")
    if cache_sample:
        print(f"-> cache_sample: timestamp:{cache_sample['timestamp']},  data_len:{len(cache_sample['data'])}")
    else:
        print("-> cache_sampLe: Cache empty!")
    print("<=====================================")

    return combined_bytes

def whisper_transribe(audio_frames: bytes(), isFake: bool = False) -> str:
    '''
    Transcribe audio frames using Whisper model.
    - audio_frames: of sample rate of 16000Hz.
    - isFake: fake transcription with random return value and time of execution.
    '''
    global LANGUAGES

    if isFake:
        time.sleep(random.randrange(6,12)/2.0)
        return ''.join([lorem.words(random.randrange(7,20)), ' '])
    
    print (f"audio_frames in bytes length: {len(audio_frames)}")

    # # TEST -> save recording to file to test audio quality
    # audio_data = sr.AudioData(audio_frames, sample_rate=SAMPLE_RATE, sample_width=2)
    # wav_data = audio_data.get_wav_data()
    # print (f"wav_data length: {len(wav_data)}")
    # audio = np.frombuffer(wav_data, np.int16).flatten()
    # print (f"wav audio sample in Int16 flattened buffer length: {len(audio)}")
    # wav_data_io = io.BytesIO(wav_data)
    # # Write wav data to the temporary file as bytes.
    # with open(f'recorded_from_mic_{sampling_count}.wav', 'w+b') as f:
    #     f.write(wav_data_io.read())
    # # END of the test

    audio = np.frombuffer(audio_frames, np.int16).flatten().astype(np.float32) / 32768.0
    print (f"audio_sample in Float32 length: {len(audio)}")

    print (f"-> sample length={len(audio) / SAMPLE_RATE} in second.")
    audio = whisper.pad_or_trim(audio)
    segments, info = model.transcribe(audio)

    # ouput only text in support languages
    transcript = ''
    if info.language in LANGUAGES:
        transcript = ''.join([segment.text for segment in segments])

    return transcript

def whisper_processing(model: WhisperModel, in_queue: Queue, socket: SocketIO):
    global backend_started
    global isFinal
    global lastTranscribedText
    global sampling_count
    global isFreshBytesAdded

    print("Transcribing from your buffers forever...\n")
    while True:
        if in_queue.empty():
            socket.sleep(0.1)
            # print ("empty input queue!")
            continue

        #TODO: how to solve states (isFinal,...), pass method as arg,...
        audio_frames = popAtMost30SecondLengthSample()

        print(F"==========> PROCESSING #{sampling_count}:")
        print (f"-> INPUT LENGTH={len(audio_frames)}")

        if len(audio_frames) == 0:
            if isFinal:
                print(f"-> Close request fired. Backend shut down!")
                print("<=====================================")
                break
            else:
                print("<=====================================")
                continue
        
        # check if fresh data present
        if not isFreshBytesAdded:
            sleep(0.05)
            print(f"-> Cached transcription={lastTranscribedText}")
            # emit cached transcription since no fresh data for this loop
            socket.emit(
                "speechData",
                buildTranscribedDataResponse())  
            print("<=====================================")
            continue

        start = time.perf_counter()
        text = whisper_transribe(audio_frames, isFake=False)
        stop = time.perf_counter()
        print(f"-> inference time: {stop - start}")
        print(f"-> transcription={text}")
        print("<=====================================")

        if text != "":
            # emit socket event to client with transcribed data
            lastTranscribedText = text
            socket.emit(
                "speechData",
                buildTranscribedDataResponse())  

        # check if finish request fired
        if isFinal:
            print(f"-> Close request fired. Backend shut down!")
            print("<=====================================")
            backend_started = False
            break                  

def isContainSpeech(message: bytearray) -> bool:
    values = [(message)[i:i + STEP_SIZE] 
                  for i in range(0, len(message), STEP_SIZE)]
    # print (values)

    is_speeches=[]
    for value in values[:-1]:
        is_speech = vad.is_speech(value, SAMPLE_RATE, MIN_CHUNK_SIZE)
        is_speeches.append(is_speech)
    # print (is_speeches)
    if any(is_speeches): return True
    else: return False

def resetNewTranscriptionJob():
    global backend_started
    global input_queue 
    global combined_bytes
    global last_sample_timestamp 
    global cache_sample 
    global isMove2NextChunk 
    global isPhraseComplete 
    global isFreshBytesAdded 
    global lastTranscribedText 
    global isFinal
    global sampling_count
    global input_queue_count

    backend_started = True                 # boolean
    while not input_queue.empty(): input_queue.get()
    combined_bytes = bytes()                # in bytes
    last_sample_timestamp = None            # timestamp
    cache_sample = None                     # {'timestamp':...,'data':...}
    isMove2NextChunk = False                # each chunk maximum 30s
    isPhraseComplete = False                # boolean
    isFreshBytesAdded = False               # boolean
    lastTranscribedText = None              # string
    isFinal = False                         # session finished fired from client
    sampling_count = 0                      # int
    input_queue_count = 0                   # int

@socketio.on('binaryAudioData')
def stream(message):
    global frames
    global input_queue
    global input_queue_count

    # message length = 4096 in bytes (=2*2048 Int16 buffersize)
    # print (f"message length={len(message['chunk'])}")

    if len(message["chunk"]) >= MIN_CHUNK_SIZE:
        if isContainSpeech(message["chunk"]):
            frames += message["chunk"]
        elif len(frames) >= MIN_CHUNK_SIZE:
            input_queue.put({'timestamp': datetime.utcnow().timestamp(), 'data': frames})

            echo_data = {'timestamp': datetime.utcnow().timestamp(), 'data_len': len(frames)}
            input_queue_count +=1
            print (f">>>input queue item #{input_queue_count}: {echo_data}")
            frames = b''           

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print(f"----> Client connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

@socketio.on('start')
def start():
    global backend_started
    global input_queue    
    global processor_thread

    print(f"backend_started: {backend_started}")
    print(f"input_queue size: {input_queue.qsize()}")

    if backend_started:
        if input_queue.qsize() > 0:
            # backend is busy, quit!
            emit("onNewJobStarted", {"started": False})
            return
    
    print("----> STARTED!")
    resetNewTranscriptionJob()

    emit("onNewJobStarted", {"started": backend_started})

    socketio.sleep(1)

    with processor_thread_lock:
        if processor_thread is None:
            processor_thread = socketio.start_background_task(
                whisper_processing,
                model, input_queue, socketio)

@socketio.on('stop')
def stop():
    global input_queue
    global CLOSE_REQUEST

    echo_data = {'timestamp': datetime.utcnow().timestamp(), 'data': CLOSE_REQUEST}
    print(f">>>stop queue item: {echo_data}")
    input_queue.put({'timestamp': datetime.utcnow().timestamp(), 'data': CLOSE_REQUEST})
    print("----> STOPPED!")

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("----> Client disconnected")
    emit("disconnect", f"user disconnected", broadcast=True)

# flask root endpoint
@app.route("/", methods=["GET"])
def welcome():
    return "Whispering something on air!"

if __name__ == "__main__":    
    # app.run(debug=True, port=5000, host="0.0.0.0")
    socketio.run(app, debug=True, port=5000, host="0.0.0.0")
