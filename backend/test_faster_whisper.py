import os
from faster_whisper import WhisperModel
import logging
import time

def print_message(msg: str):
    print(msg)
    logging.info(msg)

# read environment variables from container
MODEL_NAME = os.getenv("MODEL_NAME") if os.getenv("MODEL_NAME") else "medium"
DEVICE = os.getenv("DEVICE") if os.getenv("DEVICE") else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") if os.getenv("COMPUTE_TYPE") else "auto"

print_message(f"MODEL_NAME={MODEL_NAME}")
print_message(f"DEVICE={DEVICE}")
print_message(f"COMPUTE_TYPE={COMPUTE_TYPE}")

model = WhisperModel(MODEL_NAME, 
                     device=DEVICE,
                     compute_type=COMPUTE_TYPE,
                     download_root="models/",
                     local_files_only=True
        )

print(f"Model whisper `{MODEL_NAME}` loaded.")

last_prefix = 'Xin chào!'
last_init_prompt = ''
def fast_whisper_transcribe(audio_path: str, prefix = None, initial_prompt=None, word_timestamp=True):
    global last_prefix, last_init_prompt

    # segments, info = model.transcribe("audio.wav", beam_size=5)
    segments, info = model.transcribe(audio_path, 
                                      prefix=prefix,
                                      initial_prompt=initial_prompt,
                                      word_timestamps=word_timestamp, 
                                      beam_size=5)

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    print(f'==================================>')
    print(f'>> Transcription for {audio_path}:')
    if prefix:
        print(f'>> prefix parameter: {prefix}')
    if initial_prompt:
        print(f'>> initial_prompt param: {initial_prompt}')

    last_prefix, last_init_prompt = '', ''
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print(segment)
        last_init_prompt += segment.text
        last_prefix = segment.text
    print('<====================================')

# audio_paths = ['diemtin_AI_4-6.5s.wav', 'diemtin_AI_4-9s.wav', 'diemtin_AI_6.5-11.5s.wav', 'diemtin_AI_9-14s.wav']

audio_paths = ['diemtin_AI-0-10s.wav', 'diemtin_AI-2.5-12.5s.wav', 'diemtin_AI-5-15s.wav', 'diemtin_AI-7.5-17.5s.wav', 'diemtin_AI-10-20s.wav']

for audio_path in audio_paths:
    full_path = f'audios/{audio_path}'
    fast_whisper_transcribe(full_path, prefix=None, initial_prompt=None)

# start = time.time()
# fast_whisper_transcribe('audios/diemtin_AI-10-20s.wav', prefix=last_prefix)
# end = time.time()

# print(f'total time: {end-start}')