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

SLOT_SAMPLES = 2048 
BYTES_PER_SAMPLE = 4 # each sample of type of float32 (4 bytes) from client
SLOT_BYTES = SLOT_SAMPLES*BYTES_PER_SAMPLE

STRUCT_KEYS_SIZE = 24 # bytes
DEMO_SLOTS = 200 # with 2048 samples each slot

class Record(ctypes.Structure):
    _fields_ = [
        ('write_number', ctypes.c_uint),
        ('timestamp_microseconds', ctypes.c_ulonglong),
        ('length', ctypes.c_uint),
        ('data', ctypes.c_float * SLOT_SAMPLES),
    ]


def writer(ring, start, count):
    print ('START WRITING DATA ...')
    for i in range(start, start + count):
        data = os.urandom(SLOT_BYTES//2)
        audio = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        time_micros = int(time.time() * 10**6)
        record = Record(
            write_number=i,
            timestamp_microseconds=time_micros,
            length=len(audio))
        # Note: You can't pass 'data' to the constructor without doing an
        # additional copy to convert the bytes type to a c_ubyte * 1000. So
        # instead, the constructor will initialize the 'data' field's bytes
        # to zero, and then this assignment overwrites the data-sized part.
        record.data[:len(audio)] = audio

        try:
            ring.try_write(record)
            time.sleep(0.1)
        except ringbuffer.WaitingForReaderError:
            print('Reader is too slow, dropping %d' % i)
            continue

        if i and i % 100 == 0:
            print('Wrote %d so far' % i)

    ring.writer_done()
    print('Writer is done')


def reader(ring: ringbuffer.RingBuffer, pointer: ringbuffer.Pointer):    
    print ('START READING DATA ...')
    accumulated_buffer = []
    start = time.time()
    while True:
        # wait for writer did the first write    
        time.sleep(0.5)

        try:
            # print(f'writer index: {ring.writer.get().index}, \
            #       writer generation: {ring.writer.get().generation}')
            
            # get current counter of the writer 
            cur_writer_counter = ring.writer.get().counter
            print(f'cur_writer_counter: {cur_writer_counter}', end=', ')
            print(f'pointer.counter.value: {pointer.counter.value}')
            
            if cur_writer_counter == pointer.counter.value and ring.active.value <= 0:
                end = time.time()
                print(f'total processing time: {end - start} in sec.')
                print(f'total streaming audio length: {DEMO_SLOTS*2048/16000} in sec.')
                break

            while pointer.counter.value < cur_writer_counter:
                data = ring.blocking_read(pointer)
                record = Record.from_buffer(data)
                accumulated_buffer.extend(record.data)

            # print('============ accumulated buffer data =======')
            print(f'processing ... at {time.time()}')
            print(f'accumulated buffer with {len(accumulated_buffer)} elements')
            # print(f'accumulated buffer: {[i for i in accumulated_buffer]}')
                
            # clear the accumulated buffer after 100 iters
            accumulated_buffer.clear()
            time.sleep(1.5)
        except ringbuffer.WriterFinishedError:
            return

        # if record.write_number and record.write_number % 3 == 0:
        #     print('Reader %s saw record %d at timestamp %d with %d samples %d bytes each' %
        #           (id(pointer), record.write_number,
        #            record.timestamp_microseconds, record.length, sizeof(ctypes.c_float)))
            # print(f'Data buffer of size {sizeof(record.data)} bytes')
            # print(f'Data structure of size {sizeof(record)} bytes')
            # print(f'Data buffer: {[i for i in record.data]}')

       

    print('Reader %r is done' % id(pointer))

@socketio.on('binaryAudioData')
def stream(message):
    pass 

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print(f"----> Client connected!")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

@socketio.on('start')
def start():    
    # create a circular buffer
    ring = ringbuffer.RingBuffer(slot_bytes=SLOT_BYTES+STRUCT_KEYS_SIZE, slot_count=80)
    ring.new_writer()

    processes = [
        multiprocessing.Process(target=reader, args=(ring, ring.new_reader())),
        multiprocessing.Process(target=writer, args=(ring, 1, DEMO_SLOTS)),
    ]

    for p in processes:
        p.daemon = True
        p.start()

    for p in processes:
        # p.join(timeout=20) # terminate after 20 seconds
        p.join()

        assert not p.is_alive()
        assert p.exitcode == 0

@socketio.on('stop')
def stop():
    pass

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
