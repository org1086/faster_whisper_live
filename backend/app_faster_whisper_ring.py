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
from live_whisper_processor import LiveWhisperProcessor, TranscriptionResult
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
MIN_SLOTS = 12          # 12*2048 (samples) ~ 12*2048/16000 ~ 1.5 secs
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
LANGUAGE = os.getenv('LANGUAGE') if os.getenv('LANGUAGES') else "vi"

# Get environment variables
WHISPER_MODEL_NAME = os.getenv('WHISPER_MODEL_NAME') if os.getenv('WHISPER_MODEL_NAME') else "medium"
DEVICE = os.getenv("DEVICE") if os.getenv("DEVICE") else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") if os.getenv("COMPUTE_TYPE") else "auto"

logger.info(">>> Loading the Faster-Whisper model...")
logger.info(f">>> model_name={WHISPER_MODEL_NAME}")
logger.info(f">>> device={DEVICE}")
logger.info(f">>> compute_type={COMPUTE_TYPE}")

# model = whisper.load_model(WHISPER_MODEL_NAME)
# model = WhisperModel(WHISPER_MODEL_NAME, 
#                      device=DEVICE,
#                      compute_type=COMPUTE_TYPE,
#                      download_root="models/",
#                      local_files_only=True
#         )
# logger.info(f"Model faster-whisper `{WHISPER_MODEL_NAME}` loaded.")

processcor = LiveWhisperProcessor(model_name=WHISPER_MODEL_NAME, language=LANGUAGE, device=DEVICE, compute_type=COMPUTE_TYPE, \
                                  download_root="models/", local_files_only=True)
#-------------------------------------------------------------------------------

#------------------------Main processing functions------------------------------
# sampling_count = 0
def whisper_transribe(audio, shifted_samples: int, isFake: bool = False) -> str:
    '''
    Transcribe audio frames using Whisper model.
    - audio: tranformed streaming audio buffer of type of float32-type array.
    - isFake: fake transcription with random return value and time of execution.
    '''
    # global sampling_count
    # sampling_count += 1    

    # ======= TEST: save recording to file to test audio quality =======================
    # import speech_recognition as sr
    # import io    
    # audio_data = sr.AudioData(audio, sample_rate=SAMPLE_RATE, sample_width=2)
    # wav_data = audio_data.get_wav_data()
    # logger.info(f"wav_data length: {len(wav_data)}")
    # audio_int16 = np.frombuffer(wav_data, np.int16).flatten()
    # logger.info(f"wav audio sample in Int16 flattened buffer length: {len(audio_int16)}")
    # wav_data_io = io.BytesIO(wav_data)
    # # Write wav data to the temporary file as bytes.
    # with open(f'recorded_from_mic_{sampling_count}.wav', 'w+b') as f:
    #     f.write(wav_data_io.read())
    # ============================== END of the test ===================================

    if isFake:
        time.sleep(random.randrange(6,12)/2.0)
        return TranscriptionResult(''.join([lorem.words(random.randrange(2, 7)), ' ']), \
                                   ''.join([lorem.words(random.randrange(7, 15)), ' ']))
    
    logger.info(f"-> sample length={len(audio)/(2*SAMPLE_RATE)} in second.")
    last_confirmed_text, validating_text = processcor.transcribe(audio)

    return TranscriptionResult(last_confirmed_text, validating_text)

def whisper_processing(ring: ringbuffer.RingBuffer, pointer: ringbuffer.Pointer):   
    global is_streaming

    logger.info('START processing audio data from ringbuffer ...')
    accumulated_bytes = np.array([], np.byte)
    while True:
        try:
            # logger.info(f'writer index: {ring.writer.get().index}, \
            #       writer generation: {ring.writer.get().generation}')
            
            # get current counter of the writer 
            cur_writer_counter = ring.writer.get().counter
            cur_reader_counter = pointer.counter.value
            if cur_writer_counter - cur_reader_counter < MIN_SLOTS:
                time.sleep(0.05)
                continue
            # if cur_writer_counter - cur_reader_counter <=0: continue
            
            # logger.info(f'->cur_reader_counter: {cur_reader_counter}', end=', ')
            # logger.info(f'cur_writer_counter: {cur_writer_counter}')

            data = ring.blocking_read(pointer, cur_writer_counter - cur_reader_counter)
            accumulated_bytes= np.concatenate([Record.from_buffer(d).data for d in data])

            if not any(accumulated_bytes): continue
 
            processing_msg = f'>>> processing {len(accumulated_bytes)} bytes at {time.time()}'
            # logger.info(processing_msg)
            # logger.info(f'accumulated buffer: {[i for i in accumulated_buffer]}')
            
            # save to processing logs to file
            if is_streaming:
                f_logs.write(f'{processing_msg}\n')

            start = time.perf_counter()
            # transcribe with faster-whisper model
            text = whisper_transribe(accumulated_bytes, isFake=False)
            stop = time.perf_counter()
            logger.info(f"-> inference time: {stop - start}")
            logger.info(f"-> transcription={text}")

            # clear the accumulated buffer
            accumulated_bytes = np.array([], np.byte)
            # time.sleep(random.randint(1,4)/200)
            # burn_cpu(1000*random.randint(1,4)/0.5)

        except ringbuffer.WriterFinishedError:
            return

packages_to_ring_count = 0 
@socketio.on('binaryAudioData')
def stream(message):
    global ring, is_streaming 
    # global packages_to_ring_count, start_audio_transfer, stop_audio_transfer

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
        # logger.info('%d samples are written to the ring!' % len(audio))

        # BEGIN TEST `stream audio to ring` performance with/without heavy CPU computation
        # packages_to_ring_count +=1
        # if not packages_to_ring_count % 100: 
        #     stop_audio_transfer = time.time()
        #     logger.info(f'packages to ring: {packages_to_ring_count}')
        #     logger.info(f'total time the pushing 100 packages to ring: {stop_audio_transfer - start_audio_transfer}')
            
        #     with open("test_ring_in_perf_light_cpu_computation.txt", "w") as f:
        #         f.write(f'total time the pushing 200 packages to ring: {stop_audio_transfer - start_audio_transfer}\n')
        #     start_audio_transfer = stop_audio_transfer 
        # END OF TEST           

    except ringbuffer.WaitingForReaderError:
        logger.info('Reader is too slow, dropping audio buffer (%d) at timestamp: %.0f' % (len(message["chunk"]), time_micros))

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    logger.info(f"----> Client [{request.sid}] connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

start_audio_transfer = time.time()
@socketio.on('start')
def start():
    global ring, processor_thread, is_streaming, f_logs
    # global start_audio_transfer
    
    f_logs = open('processing_logs.txt', 'w')

    logger.info('>>> on_start event from client fired!')
    is_streaming = True

    # start_audio_transfer = time.time()

    if not processor_thread:
        # init processing thread
        processor_thread = threading.Thread(target=whisper_processing, args=(ring, ring.new_reader()))
        logger.info(f'>>> initialized processing thread {processor_thread}')
        processor_thread.start()
        # processor_thread.join()
    else:
        logger.info(f'>>> existing processor_thread: {processor_thread}')
        logger.info(f'>>> existing processor_thread.is_alive: {processor_thread.is_alive()}')

        if not processor_thread.is_alive():
            processor_thread.start()
            logger.info(f'>>> started existing processor_thread: {processor_thread}')
            # processor_thread.join()

@socketio.on('stop')
def stop():
    global is_streaming, f_logs 

    is_streaming = False    
    f_logs.close()

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
