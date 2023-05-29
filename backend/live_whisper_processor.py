from ringbuffer import ringbuffer
import faster_whisper
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word
from app_faster_whisper_ring import print_message


class LiveWhisperProcessorBase:

    def __init__(self, model_name:str, device: str, compute_type: str, \
                   download_root: str, local_files_only: bool):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.local_files_only = local_files_only

        self.load_model()

        self.confirmed_words = []       # array of Word object
        self.validating_words = []      # array of Word object       

    def load_model(self):
        raise NotImplemented("must be implemented in the child class")

    def transribe(self, audio):
        raise NotImplemented("must be implemented in the child class")
    
    

class LiveWhisperProcessor:
    pass