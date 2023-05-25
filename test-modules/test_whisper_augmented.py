from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("Scrya/whisper-medium-vi-augmented")

model = AutoModelForSpeechSeq2Seq.from_pretrained("Scrya/whisper-medium-vi-augmented")