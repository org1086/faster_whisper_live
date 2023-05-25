import os
from faster_whisper import WhisperModel
import logging

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
                     download_root="models/"
        )

print(f"Model whisper `{MODEL_NAME}` loaded.")

# segments, info = model.transcribe("audio.wav", beam_size=5)
segments, info = model.transcribe("diemtin_AI.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))