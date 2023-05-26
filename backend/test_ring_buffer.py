#!/usr/bin/env python3

"""Simple example with ctypes.Structures."""

import ctypes
from ctypes import sizeof
import multiprocessing
import os
import random
import time
import numpy as np

import os
import time
import threading
from time import sleep
from datetime import datetime, timedelta
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from queue import Queue
from engineio.payload import Payload
from tempfile import NamedTemporaryFile
import logging
import io
import warnings

from ringbuffer import ringbuffer

Payload.max_decode_packets = 500
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
warnings.filterwarnings("ignore")

async_mode = 'eventlet'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# min buffer length of interest
MIN_CHUNK_SIZE = 480                    # in bytes

# audio streaming in progress
is_streaming = False

f_processing_log = open('processing_logs.txt', 'w')
#------------------------ringbuffer setups------------------------------------
SLOT_SAMPLES = 2048 
BYTES_PER_SAMPLE = 4 # each sample of type of float32 (4 bytes) from client
SLOT_BYTES = SLOT_SAMPLES*BYTES_PER_SAMPLE

STRUCT_KEYS_SIZE = 16 # bytes
DEMO_SLOTS = 200 # with 2048 samples each slot

class Record(ctypes.Structure):
    _fields_ = [
        ('timestamp_microseconds', ctypes.c_ulonglong),
        ('length', ctypes.c_uint),
        ('data', ctypes.c_float * SLOT_SAMPLES),
    ]


# create a circular buffer
ring = ringbuffer.RingBuffer(slot_bytes=SLOT_BYTES+STRUCT_KEYS_SIZE, slot_count=80)
ring.new_writer()

# thread for processing audio stream
processor_thread = None
#-------------------------------------------------------------------------------

def writer(ring, start, count):
    print ('START receiving audio streaming data...')
    for i in range(start, start + count):
        data = os.urandom(SLOT_BYTES//2)
        audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        time_micros = int(time.time() * 10**6)
        record = Record(
            timestamp_microseconds=time_micros,
            length=len(audio))
        # Note: You can't pass 'data' to the constructor without doing an
        # additional copy to convert the bytes type to a c_ubyte * 1000. So
        # instead, the constructor will initialize the 'data' field's bytes
        # to zero, and then this assignment overwrites the data-sized part.
        record.data[:len(audio)] = audio

        try:
            ring.try_write(record)
            time.sleep(0.05)
        except ringbuffer.WaitingForReaderError:
            print('Reader is too slow, dropping %d' % i)
            continue

    ring.writer_done()
    print('Writer is done')

def burn_cpu(milliseconds):
    start = now = time.time()
    end = start + milliseconds / 1000
    while True:
        now = time.time()
        if now >= end:
            break
        for i in range(100):
            random.random() ** 1 / 2

def processing_audio(ring: ringbuffer.RingBuffer, pointer: ringbuffer.Pointer):   
    global is_streaming

    print ('START processing audio data from ringbuffer ...')
    accumulated_buffer = []
    while True:
        try:
            # print(f'writer index: {ring.writer.get().index}, \
            #       writer generation: {ring.writer.get().generation}')
            
            # get current counter of the writer 
            cur_writer_counter = ring.writer.get().counter
            cur_reader_counter = pointer.counter.value
            if cur_reader_counter < cur_writer_counter:
                print(f'->cur_reader_counter: {cur_reader_counter}', end=', ')
                print(f'cur_writer_counter: {cur_writer_counter}')

                data = ring.blocking_read(pointer, cur_writer_counter - cur_reader_counter)
                records = [Record.from_buffer(d) for d in data]
                [accumulated_buffer.extend(record.data) for record in records]

            # print('============ accumulated buffer data =======')
            processing_msg = ''
            if len(accumulated_buffer):
                processing_msg = f'>>> processing {len(accumulated_buffer)} samples at {time.time()}'
                print(processing_msg)
                # print(f'accumulated buffer: {[i for i in accumulated_buffer]}')
            
            # save to processing logs to file
            if is_streaming:
                f_processing_log.write(f'{processing_msg}\n')

            # clear the accumulated buffer after 100 iters
            accumulated_buffer.clear()
            # time.sleep(random.randint(1,4)/2)
            burn_cpu(1000*random.randint(1,4)/2)
        except ringbuffer.WriterFinishedError:
            return

        # if record.write_number and record.write_number % 3 == 0:
        #     print('Reader %s saw record %d at timestamp %d with %d samples %d bytes each' %
        #           (id(pointer), record.write_number,
        #            record.timestamp_microseconds, record.length, sizeof(ctypes.c_float)))
        #     print(f'Data buffer of size {sizeof(record.data)} bytes')
        #     print(f'Data buffer: {[i for i in record.data]}')       

    print('Reader %r is done' % id(pointer))

@socketio.on('binaryAudioData')
def stream(message):
    global ring, is_streaming

    if not is_streaming: return

    msg_length = len(message["chunk"])
    if msg_length < MIN_CHUNK_SIZE: return
              
    audio = np.frombuffer(message["chunk"], np.int16).flatten().astype(np.float32) / 32768.0
    time_micros = int(time.time() * 10**6)
    record = Record(
        timestamp_microseconds=time_micros,
        length=len(audio))
    # Note: You can't pass 'data' to the constructor without doing an
    # additional copy to convert the bytes type to a c_ubyte * 1000. So
    # instead, the constructor will initialize the 'data' field's bytes
    # to zero, and then this assignment overwrites the data-sized part.
    record.data[:len(audio)] = audio

    try:
        ring.try_write(record)
        # print('%d samples are written to the ring!' % len(audio))
    except ringbuffer.WaitingForReaderError:
        print('Reader is too slow, dropping audio buffer (%d}) at timestamp: %.0f' % (len(audio), time_micros))

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print(f"----> Client connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

@socketio.on('start')
def start():
    global ring, processor_thread, is_streaming, f_processing_log
    
    f_processing_log = open('processing_logs.txt', 'w')

    print('on_start event from client fired!')
    is_streaming = True

    if not processor_thread:
        # init processing thread
        processor_thread = threading.Thread(target=processing_audio, args=(ring, ring.new_reader()))
        print(f'initialized processing thread {processor_thread}')
        processor_thread.start()
        # processor_thread.join()
    else:
        print (f'existing processor_thread: {processor_thread}')
        print (f'existing processor_thread.is_alive: {processor_thread.is_alive()}')

        if not processor_thread.is_alive():
            processor_thread.start()
            print (f'started existing processor_thread: {processor_thread}')
            # processor_thread.join()

@socketio.on('stop')
def stop():
    global is_streaming

    is_streaming = False
    
    f_processing_log.close()

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("----> Client disconnected")
    emit("disconnect", f"user disconnected", broadcast=True)

# flask root endpoint
@app.route("/", methods=["GET"])
def welcome():
    return "Whispering something on air!"

def main():
    socketio.run(app, debug=False, port=5000, host="0.0.0.0")

if __name__ == '__main__':
    main()
