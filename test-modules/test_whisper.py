import whisper
import time
import os
from os import path

audio_test =  path.join(os.path.dirname(os.path.realpath(__file__)), '../audios/diemtin_AI-30s.wav')

model = whisper.load_model("tiny")
start_time = time.time()
print (f"start time: {start_time}")
result = model.transcribe(audio_test)
end_time = time.time()
print (f"end time: {end_time}")
print(result)
print (f"time of transcribing: {end_time - start_time}")