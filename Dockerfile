FROM ghcr.io/opennmt/ctranslate2:latest-ubuntu20.04-cuda11.2

# clear entry point from original image
ENTRYPOINT []

WORKDIR /app
COPY . .
RUN python3 -m pip install -r requirements.txt

CMD ["python3", "backend.app_faster_whisper_ring.py"]
