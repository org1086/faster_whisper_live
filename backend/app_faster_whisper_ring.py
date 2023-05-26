#!/usr/bin/env python3

import os
import time
import ctypes
from ctypes import sizeof
import random
import numpy as np
import logging
import warnings
import threading
from time import sleep
from datetime import datetime, timedelta
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from engineio.payload import Payload
from lorem_text import lorem

import whisper
from faster_whisper import WhisperModel
from ringbuffer import ringbuffer

# ------------------------ Helper functions ---------------------------------------
def print_message(msg: str, end='\n'):
    print(msg, end=end)
    logging.info(msg)

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

async_mode = 'eventlet'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# min buffer length of interest
MIN_CHUNK_SIZE = 480                    # in bytes
SAMPLE_RATE = 16000                     # hertz

# audio streaming in progress
is_streaming = False

f_logs = open('processing_logs.txt', 'w')
#--------------------------------------------------------------------------------

#------------------------ringbuffer setups---------------------------------------
SLOT_BYTES = 4096
SLOT_COUNT = 160        # ring_buffer with 160*4096 in bytes
                        # or 160*4096/2 (samples) or 160*4096/2/16000 ~ 20 secs
STRUCT_KEYS_SIZE = 16   # bytes
MIN_SLOTS = 8           # 8*2048 (samples) ~ 8*2048/16000 ~ 1.024 secs
DEMO_SLOTS = 200 # with 2048 samples each slot

class Record(ctypes.Structure):
    _fields_ = [
        ('timestamp_microseconds', ctypes.c_ulonglong),
        ('length', ctypes.c_uint),
        ('data', ctypes.c_ubyte * SLOT_BYTES),
    ]

# create a circular buffer
ring = ringbuffer.RingBuffer(slot_bytes=SLOT_BYTES+STRUCT_KEYS_SIZE, slot_count=SLOT_COUNT)
ring.new_writer()

# thread for processing audio stream
processor_thread = None
#-------------------------------------------------------------------------------

#------------------------Whisper model setups-----------------------------------
# get support language set for output
LANGUAGES = os.getenv('LANGUAGES') if os.getenv('LANGUAGES') else "vi,en"
LANGUAGES  = [item.strip() for item in LANGUAGES.split(',')]

# Get environment variables
WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL_NAME') if os.getenv('WHISPER_MODEL_NAME') else "medium"
DEVICE = os.getenv("DEVICE") if os.getenv("DEVICE") else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") if os.getenv("COMPUTE_TYPE") else "auto"

print_message(">>> Loading the Faster-Whisper model...")
print_message(f">>> model_name={WHISPER_MODEL_NAME}")
print_message(f">>> device={DEVICE}")
print_message(f">>> compute_type={COMPUTE_TYPE}")

# model = whisper.load_model(WHISPER_MODEL_NAME)
model = WhisperModel(WHISPER_MODEL_NAME, 
                     device=DEVICE,
                     compute_type=COMPUTE_TYPE,
                     download_root="models/",
                     local_files_only=True
        )
print_message(f"Model faster-whisper `{WHISPER_MODEL_NAME}` loaded.")
#-------------------------------------------------------------------------------

#------------------------Main processing functions------------------------------
sampling_count = 0
def whisper_transribe(audio, isFake: bool = False) -> str:
    '''
    Transcribe audio frames using Whisper model.
    - audio: tranformed streaming audio buffer of type of float32-type array.
    - isFake: fake transcription with random return value and time of execution.
    '''
    global LANGUAGES
    global sampling_count
    sampling_count += 1    

    # TEST: save recording to file to test audio quality
    import speech_recognition as sr
    import io    
    audio_data = sr.AudioData(audio, sample_rate=SAMPLE_RATE, sample_width=2)
    wav_data = audio_data.get_wav_data()
    print_message(f"wav_data length: {len(wav_data)}")
    audio_int16 = np.frombuffer(wav_data, np.int16).flatten()
    print (f"wav audio sample in Int16 flattened buffer length: {len(audio_int16)}")
    wav_data_io = io.BytesIO(wav_data)
    # Write wav data to the temporary file as bytes.
    with open(f'recorded_from_mic_{sampling_count}.wav', 'w+b') as f:
        f.write(wav_data_io.read())
    # END of the test

    if isFake:
        time.sleep(random.randrange(6,12)/2.0)
        return ''.join([lorem.words(random.randrange(7,20)), ' '])
    
    audio_float32 = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
    print_message(f"-> sample length={len(audio_float32) / SAMPLE_RATE} in second.")
    # audio = whisper.pad_or_trim(audio)
    segments, info = model.transcribe(audio_float32)

    # ouput only text in support languages
    transcript = ''
    if info.language in LANGUAGES:
        transcript = ''.join([segment.text for segment in segments])

    return transcript

def whisper_processing(ring: ringbuffer.RingBuffer, pointer: ringbuffer.Pointer):   
    global is_streaming

    print ('START processing audio data from ringbuffer ...')
    accumulated_bytes = np.array([], np.byte)
    while True:
        try:
            # print_message(f'writer index: {ring.writer.get().index}, \
            #       writer generation: {ring.writer.get().generation}')
            
            # get current counter of the writer 
            cur_writer_counter = ring.writer.get().counter
            cur_reader_counter = pointer.counter.value
            if cur_writer_counter - cur_reader_counter < MIN_SLOTS:
                time.sleep(0.05)
                continue
            
            print_message(f'->cur_reader_counter: {cur_reader_counter}', end=', ')
            print_message(f'cur_writer_counter: {cur_writer_counter}')

            data = ring.blocking_read(pointer, cur_writer_counter - cur_reader_counter)
            accumulated_bytes= np.concatenate([Record.from_buffer(d).data for d in data])

            if not any(accumulated_bytes): continue
 
            processing_msg = f'>>> processing {len(accumulated_bytes)} bytes at {time.time()}'
            print_message(processing_msg)
            # print_message(f'accumulated buffer: {[i for i in accumulated_buffer]}')
            
            # save to processing logs to file
            if is_streaming:
                f_logs.write(f'{processing_msg}\n')

            start = time.perf_counter()
            # transcribe with faster-whisper model
            text = whisper_transribe(accumulated_bytes, isFake=False)
            stop = time.perf_counter()
            print_message(f"-> inference time: {stop - start}")
            print_message(f"-> transcription={text}")

            # clear the accumulated buffer
            accumulated_bytes = np.array([], np.byte)
            time.sleep(random.randint(1,4)/200)
            # burn_cpu(1000*random.randint(1,4)/2)

        except ringbuffer.WriterFinishedError:
            return

@socketio.on('binaryAudioData')
def stream(message):
    global ring, is_streaming

    if not is_streaming: return

    msg_length = len(message["chunk"])
    if msg_length < MIN_CHUNK_SIZE: return
              
    time_micros = int(time.time() * 10**6)
    record = Record(
        timestamp_microseconds=time_micros,
        length=msg_length)
    # Note: You can't pass 'data' to the constructor without doing an
    # additional copy to convert the bytes type to a c_ubyte * 1000. So
    # instead, the constructor will initialize the 'data' field's bytes
    # to zero, and then this assignment overwrites the data-sized part.
    record.data[:msg_length] = message["chunk"]

    try:
        ring.try_write(record)
        # print_message('%d samples are written to the ring!' % len(audio))
    except ringbuffer.WaitingForReaderError:
        print_message('Reader is too slow, dropping audio buffer (%d) at timestamp: %.0f' % (len(audio), time_micros))

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print_message(f"----> Client [{request.sid}] connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

@socketio.on('start')
def start():
    global ring, processor_thread, is_streaming, f_logs
    
    f_logs = open('processing_logs.txt', 'w')

    print_message('>>> on_start event from client fired!')
    is_streaming = True

    if not processor_thread:
        # init processing thread
        processor_thread = threading.Thread(target=whisper_processing, args=(ring, ring.new_reader()))
        print_message(f'>>> initialized processing thread {processor_thread}')
        processor_thread.start()
        # processor_thread.join()
    else:
        print (f'>>> existing processor_thread: {processor_thread}')
        print (f'>>> existing processor_thread.is_alive: {processor_thread.is_alive()}')

        if not processor_thread.is_alive():
            processor_thread.start()
            print (f'>>> started existing processor_thread: {processor_thread}')
            # processor_thread.join()

@socketio.on('stop')
def stop():
    global is_streaming

    is_streaming = False
    
    f_logs.close()

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print_message("----> Client disconnected")
    emit("disconnect", f"user disconnected", broadcast=True)

# flask root endpoint
@app.route("/", methods=["GET"])
def welcome():
    return "Whispering something on air!"

def main():
    socketio.run(app, debug=False, port=5000, host="0.0.0.0")

if __name__ == '__main__':
    main()
