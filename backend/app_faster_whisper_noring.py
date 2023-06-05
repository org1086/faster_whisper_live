#!/usr/bin/env python3

import os
import time
from time import sleep
from datetime import datetime, timedelta
import ctypes
from ctypes import sizeof
import random
import io
import numpy as np
import logging
import warnings
import threading
import eventlet
from threading import Lock
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from engineio.payload import Payload
from lorem_text import lorem
import webrtcvad
import whisper
from whisper import Whisper
from faster_whisper import WhisperModel
from queue import Queue

from logger import initialize_logger

# init logger
logger = initialize_logger(__name__, logging.DEBUG)

# ------------------------ Helper functions ---------------------------------------

def burn_cpu(milliseconds):
    start = now = time.time()
    end = start + milliseconds / 1000
    while True:
        now = time.time()
        if now >= end:
            break
        for i in range(100):
            random.random() ** 1 / 2
#--------------------------------------------------------------------------------

# ------------------------ Flask app and socketio setups ------------------------
Payload.max_decode_packets = 50
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})
warnings.filterwarnings("ignore")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
# this patch is to fix emitting socket event from child thread
# remove it since it cause audio streaming very slow (lost audio package)
# eventlet.monkey_patch()

vad = webrtcvad.Vad()
vad.set_mode(3)
frames = b''

SAMPLE_RATE = 16000                     # hertz

# check if speech exists
MIN_CHUNK_SIZE = 480                    # in bytes
STEP_SIZE = 960                         # in bytes

# phrase complete and max buffer size (max_length=30s)
PHRASE_TIMEOUT = 7                      # in seconds
BYTE_PER_SAMPLE = 2
THIRTY_SECS_SIZE = 30*BYTE_PER_SAMPLE*SAMPLE_RATE     # 30 seconds
MIN_SAMPLE_LENGTH = 1*BYTE_PER_SAMPLE*SAMPLE_RATE     # 1 seconds

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

# audio streaming in progress
is_streaming = False

# thread for processing audio stream
processor_thread_lock = Lock()
processor_thread = None
#-------------------------------------------------------------------------------

#------------------------Whisper model setups-----------------------------------
# get support language set for output
LANGUAGES = os.getenv('LANGUAGES') if os.getenv('LANGUAGES') else "vi,en"
LANGUAGES  = [item.strip() for item in LANGUAGES.split(',')]

# Get environment variables
WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL_NAME') if os.getenv('WHISPER_MODEL_NAME') else "tiny"
DEVICE = os.getenv("DEVICE") if os.getenv("DEVICE") else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") if os.getenv("COMPUTE_TYPE") else "auto"

logger.info(">>> Loading the Faster-Whisper model...")
logger.info(f">>> model_name={WHISPER_MODEL_NAME}")
logger.info(f">>> device={DEVICE}")
logger.info(f">>> compute_type={COMPUTE_TYPE}")

# model = whisper.load_model(WHISPER_MODEL_NAME)
model = WhisperModel(WHISPER_MODEL_NAME, 
                     device=DEVICE,
                     compute_type=COMPUTE_TYPE,
                     download_root="models/",
                     local_files_only=True
        )
logger.info(f"Model faster-whisper `{WHISPER_MODEL_NAME}` loaded.")

#-------------------------------------------------------------------------------

#------------------------Main processing functions------------------------------
def buildTranscribedDataResponse():
    global lastTranscribedText
    global isMove2NextChunk
    global isPhraseComplete

    return {
        'data': lastTranscribedText, 
        'isMove2NextChunk': isMove2NextChunk,
        'isPhraseComplete': isPhraseComplete
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
    global sampling_count

    # =========== SAMPLING =============
    isFreshBytesAdded = False
    sampling_count +=1
    logger.info(f"==========> SAMPLING #{sampling_count}:")
    
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
        logger.info(f"<-...cache sample (len={len(combined_bytes)}) added!")

    if not last_sample_timestamp:
        first_sample = input_queue.get()

        last_sample_timestamp = first_sample['timestamp']
        combined_bytes = first_sample['data']
        isFreshBytesAdded = True
        logger.info(f"-> 1st sample (len={len(first_sample['data'])}) for a new chunk added!")

    while (not input_queue.empty()) and (len(combined_bytes) < THIRTY_SECS_SIZE):
        current_sample = input_queue.get()
        current_sample_timestamp = current_sample['timestamp']
        current_sample_data = current_sample['data']
        
        time_gap = datetime.utcfromtimestamp(current_sample_timestamp) \
                    - datetime.utcfromtimestamp(last_sample_timestamp)
        logger.info(f"-> time_gap={time_gap}")
        
        if time_gap < timedelta(seconds=PHRASE_TIMEOUT):
            if len(combined_bytes) + len(current_sample_data) <= THIRTY_SECS_SIZE:
                
                logger.info("-> +++appending to combined_bytes....")
                logger.info(f"-> current bytes: {len(combined_bytes)}")
                logger.info(f"-> adding bytes: {len(current_sample_data)}")
                logger.info(f"-> resulted bytes: {len(combined_bytes) + len(current_sample_data)}")

                last_sample_timestamp = current_sample_timestamp
                combined_bytes += current_sample_data
                isFreshBytesAdded = True
            else:
                cache_sample = current_sample
                last_sample_timestamp = None
                isMove2NextChunk = True
                logger.info("->...caching, move to next chunk....")
                break
        else:
            cache_sample = current_sample
            last_sample_timestamp = None
            isPhraseComplete = True
            logger.info("->...caching, finish a phrase, new line....")
            break

    logger.info(f"-> Sampling finalizing...")
    if not isFreshBytesAdded:
        logger.info("-> No fresh data added!")
    logger.info(f"-> isMove2NextChunk: {isMove2NextChunk}")
    logger.info(f"-> isPhraseComplete: {isPhraseComplete}")
    logger.info(f"-> isFinal: {isFinal}")
    logger.info(f"-> OUTPUT LENGTH: {len(combined_bytes)}")
    if cache_sample:
        logger.info(f"-> cache_sample: timestamp:{cache_sample['timestamp']},  data_len:{len(cache_sample['data'])}")
    else:
        logger.info("-> cache_sampLe: Cache empty!")
    logger.info("<=====================================")

    return combined_bytes

def whisper_transribe(audio_frames: bytes(), isFake: bool = False) -> str:
    '''
    Transcribe audio frames using Whisper model.
    - audio_frames: of sample rate of 16000Hz.
    - isFake: fake transcription with random return value and time of execution.
    '''
    global LANGUAGES

    if isFake:
        socketio.sleep(random.randrange(6,12)/2.0)
        return ''.join([lorem.words(random.randrange(7,20)), ' '])
    
    logger.info(f"audio_frames in bytes length: {len(audio_frames)}")

    # # TEST -> save recording to file to test audio quality
    # audio_data = sr.AudioData(audio_frames, sample_rate=SAMPLE_RATE, sample_width=2)
    # wav_data = audio_data.get_wav_data()
    # logger.info(f"wav_data length: {len(wav_data)}")
    # audio = np.frombuffer(wav_data, np.int16).flatten()
    # logger.info(f"wav audio sample in Int16 flattened buffer length: {len(audio)}")
    # wav_data_io = io.BytesIO(wav_data)
    # # Write wav data to the temporary file as bytes.
    # with open(f'recorded_from_mic_{sampling_count}.wav', 'w+b') as f:
    #     f.write(wav_data_io.read())
    # # END of the test

    audio = np.frombuffer(audio_frames, np.int16).flatten().astype(np.float32) / 32768.0
    logger.info(f"audio_sample in Float32 length: {len(audio)}")

    logger.info(f"-> sample length={len(audio) / SAMPLE_RATE} in second.")
    # audio = whisper.pad_or_trim(audio)
    segments, info = model.transcribe(audio)
    segments = list(segments)                   # convert segment iterator to list

    # ouput only text in support languages
    transcript = ''
    if info.language in LANGUAGES:
        transcript = ''.join([segment.text for segment in segments])

    return transcript

def whisper_processing(model: WhisperModel, in_queue: Queue, socket: SocketIO):
    global is_streaming
    global lastTranscribedText
    global sampling_count
    global isFreshBytesAdded

    logger.info("Transcribing from your buffers forever...\n")
    while True:
        if in_queue.empty():
            socket.sleep(0.1)
            # logger.info("empty input queue!")
            continue

        #TODO: how to solve states (isFinal,...), pass method as arg,...
        audio_frames = popAtMost30SecondLengthSample()

        logger.info(F"==========> PROCESSING #{sampling_count}:")
        logger.info(f"-> INPUT LENGTH={len(audio_frames)}")

        if len(audio_frames) <= MIN_SAMPLE_LENGTH:
            logger.info(f"audio sample length {len(audio_frames)} too short, ignore!")
            logger.info("<=====================================")
            continue
        
        # check if fresh data present
        if not isFreshBytesAdded:
            socket.sleep(0.06)
            logger.info(f"-> Cached transcription={lastTranscribedText}")
            # emit cached transcription since no fresh data for this loop
            socket.emit(
                "speechData",
                buildTranscribedDataResponse())  
            logger.info("<=====================================")
            continue

        start = time.perf_counter()
        text = whisper_transribe(audio_frames, isFake=False)
        stop = time.perf_counter()
        logger.info(f"-> inference time: {stop - start}")
        logger.info(f"-> transcription={text}")
        logger.info("<=====================================")

        if text != "":
            # emit socket event to client with transcribed data
            lastTranscribedText = text
            socket.emit(
                "speechData",
                buildTranscribedDataResponse())              

def isContainSpeech(message: bytearray) -> bool:
    values = [(message)[i:i + STEP_SIZE] 
                  for i in range(0, len(message), STEP_SIZE)]
    # logger.info(values)

    is_speeches=[]
    for value in values[:-1]:
        is_speech = vad.is_speech(value, SAMPLE_RATE, MIN_CHUNK_SIZE)
        is_speeches.append(is_speech)
    # logger.info(is_speeches)
    if any(is_speeches): return True
    else: return False

#-------------------------------------------------------------------------------

packages_to_ring_count = 0 
@socketio.on('binaryAudioData')
def stream(message):
    global is_streaming
    global frames
    global input_queue
    global input_queue_count

    if not is_streaming: return

    # message length = 4096 in bytes (=2*2048 Int16 buffersize)
    # logger.info(f"message length={len(message['chunk'])}")

    if len(message["chunk"]) >= MIN_CHUNK_SIZE:
        if isContainSpeech(message["chunk"]):
            frames += message["chunk"]
        elif len(frames) >= MIN_CHUNK_SIZE:
            input_queue.put({'timestamp': datetime.utcnow().timestamp(), 'data': frames})

            echo_data = {'timestamp': datetime.utcnow().timestamp(), 'data_len': len(frames)}
            input_queue_count +=1
            logger.info(f">>>input queue item #{input_queue_count}: {echo_data}")
            frames = b''     

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    logger.info(f"----> Client [{request.sid}] connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

start_audio_transfer = time.time()
@socketio.on('start')
def start():
    global processor_thread_lock, processor_thread, is_streaming, socketio    
    global input_queue

    logger.info('>>> on_start event from client fired!')
    is_streaming = True

    with processor_thread_lock:
        if not processor_thread:
            processor_thread = socketio.start_background_task(
                whisper_processing,
                model, input_queue, socketio)
        else:
            logger.info(f'>>> existing processor_thread: {processor_thread}')
            logger.info(f'>>> existing processor_thread.is_alive: {processor_thread.is_alive()}')

            if not processor_thread.is_alive():
                processor_thread.start()
                logger.info(f'>>> started existing processor_thread: {processor_thread}')
                # processor_thread.join()
        
    logger.info("----> STARTED!")

@socketio.on('stop')
def stop():
    global is_streaming 

    is_streaming = False   
    logger.info("----> STOPPED!")

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    logger.info("----> Client disconnected")
    emit("disconnect", f"user disconnected", broadcast=True)

# flask root endpoint
@app.route("/", methods=["GET"])
def welcome():
    return "Whispering something on air!"

def main():
    socketio.run(app, debug=False, port=5000, host="0.0.0.0")

if __name__ == '__main__':
    main()
