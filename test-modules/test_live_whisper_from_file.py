import pyaudio
import webrtcvad
import whisper
import numpy as np
import threading
import copy
import time
import os
from os import path
from sys import exit
import contextvars
from queue import Queue

import wave
# import soundfile as sf

audio_test =  path.join(os.path.dirname(os.path.realpath(__file__)), '../audios/eu.wav')

class LiveWhisper():
    exit_event = threading.Event()
    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name

    def stop(self):
        """stop the asr process"""
        LiveWhisper.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=LiveWhisper.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue,))
        self.asr_process.start()
        time.sleep(1)  # start vad after asr model is loaded
        self.vad_process = threading.Thread(target=LiveWhisper.vad_process, args=(
            self.device_name, self.asr_input_queue,))
        self.vad_process.start()

    def vad_process(device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        # audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 50

        # microphones = LiveWav2Vec2.list_microphones(audio)
        # selected_input_device_id = LiveWav2Vec2.get_input_device_id(
        #     device_name, microphones)

        # stream = audio.open(input_device_index=selected_input_device_id,
        #                     format=FORMAT,
        #                     channels=CHANNELS,
        #                     rate=RATE,
        #                     input=True,
        #                     frames_per_buffer=CHUNK)

        # stream from file
        wf = wave.open(audio_test, 'rb')
        # stream = audio.open(
        #     format = FORMAT,
        #     channels = CHANNELS,
        #     rate = RATE,
        #     # input = True
        #     output= True
        # )

        frames = b''
        while True:
            if LiveWhisper.exit_event.is_set():
                break
            
            # get data chunk from wave file object
            frame = wf.readframes(CHUNK)

            if frame != b'':

                is_speech = vad.is_speech(frame, RATE)
                if is_speech:
                    frames += frame
                    # print(frames)
                else:
                    if len(frames) > 1:
                        asr_input_queue.put(frames)
                    frames = b''
        # stream.stop_stream()
        # stream.close()
        # audio.terminate()

    def asr_process(model_name, in_queue, output_queue):
        model =  whisper.load_model(model_name)

        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break
            #convert bytes (audio_frames) to array (float64_buffer)
            audio = np.frombuffer(audio_frames, np.int16).flatten().astype(np.float32) / 32768.0

            print (f"sample_length={len(audio) / 16000} in second.")
            audio = whisper.pad_or_trim(audio)
            print (f"sample_length_pad={ len(audio) / 16000} in second.")

            start = time.perf_counter()
            text = model.transcribe(audio)
            inference_time = time.perf_counter()-start
            sample_length = len(audio) / 16000  # length in sec
            if text['text'] != "":
                output_queue.put([text['text'],sample_length,inference_time])

    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()

if __name__ == "__main__":
    print("Live Whisper")

    asr = LiveWhisper("tiny")

    asr.start()

    try:
        while True:
            text,sample_length,inference_time = asr.get_last_text()
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{text}")

    except KeyboardInterrupt:
        asr.stop()
        exit()
