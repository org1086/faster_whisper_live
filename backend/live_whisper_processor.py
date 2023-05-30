import logging
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word, Segment
from logger import initialize_logger
import numpy as np
from typing import NamedTuple, Iterable
import collections

logger = initialize_logger(__name__, logging.DEBUG)

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max):
        self.max = size_max
        self.data = []
        self.cur = 0

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def extend(self, xs):
            """ Extend multiple elements to the ring. """

            # WARN: update current position beforehand
            self.cur = (self.cur + len(xs)) % self.max

            # update data to the ring
            if len(xs) >= self.max:
                update_data = xs[-self.max:]
                self.data[self.cur:] = update_data[:(self.max-self.cur)]
                self.data[:self.cur] = update_data[(self.max-self.cur):]
            else:
                update_indexes = [ i%self.max for i in range(self.cur + self.max - len(xs), self.cur + self.max)]
                for i,j in zip(update_indexes,xs):
                    self.data[i] = j

        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """ Append an element at the end of the buffer. """
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            self.__class__ = self.__Full
        else: 
            self.cur +=1
    
    def extend(self, xs):
        """ Extend multiple elements to the buffer. """
        if self.max - len(self.data) >= len(xs):
            self.data.extend(xs)
            if len(self.data) == self.max:
                self.cur = 0
                self.__class__ = self.__Full
            else:
                self.cur += len(xs)
        else:
            # add 0s to fill the buffer
            self.data.extend([0.0]*(self.max - self.cur))

            # switch this class to __Full
            self.__class__ = self.__Full
            
            # call extend method of the __Full class
            self.extend(xs)

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

class TranscriptionResult(NamedTuple):
    last_confirmed: str
    validating: str

class BufferWindow(NamedTuple):
    position: int
    length: int

class LiveWhisperProcessorBase:

    def __init__(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")    

    def load_model(self):
        raise NotImplemented("must be implemented in the child class")

    def transribe(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")

    def processing(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    

class LiveWhisperProcessor(LiveWhisperProcessorBase):
    def __init__(self, 
                 model_name:str = 'tiny', 
                 language: str = 'vi', 
                 device: str = 'cpu', 
                 compute_type: str = 'auto',
                 download_root: str = 'models/', 
                 local_files_only: bool = True,
                 window_samples: int = 160000                 
                ):
        
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.window_samples = window_samples

        # ring buffer to store buffers to process
        self.audio_buffer = RingBuffer(self.window_samples)

        self.counter = 0                    # sample counter current sampling time

        self.previous_window = BufferWindow(0,0)
        self.current_window = BufferWindow(0,0)

        self.confirmed_words = []           # array of Word object
        self.validating_words = []          # array of Word object  

        self.load_model()
    
    def load_model(self):
        self.model = WhisperModel(self.model_name, 
                            device=self.device,
                            compute_type=self.compute_type,
                            download_root=self.download_root,
                            local_files_only=self.local_files_only
                    )
        logger.info(f"Model faster-whisper `{self.model_name}` loaded.")

    def transcribe(self, audio):
        '''
        Transcribe audio frames using Whisper model.
        - audio: tranformed streaming audio buffer of type of float32-type array.
        '''
        new_audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0

        self.previous_window = self.current_window

        # push to the overridable buffer of the processor
        self.audio_buffer.extend(new_audio)
        self.counter += len(new_audio)
        
        if self.counter <= self.window_samples:
            self.current_window.position = 0
            self.current_window.length = self.counter
        else:
            self.current_window.position = self.counter - self.window_samples
            self.current_window.length = self.window_samples
        


        # think how to calculate confirmed window, overlapping window, and new window 
        

        # segments, _ = self.model.transcribe(audio_float32, language = self.language)

        # return self.processing(segments)

    def processing(self, new_segments: Iterable[Segment]) -> TranscriptionResult:
        pass

if __name__ == "__main__":
    my_ring = RingBuffer(10)

    print(my_ring.data)
    print(my_ring.cur)

    my_ring.append(1)
    print(my_ring.data)
    print(my_ring.cur)

    my_ring.extend(range(2,5))
    print(my_ring.data)
    print(my_ring.cur)

    my_ring.append(5)
    print(my_ring.data)
    print(my_ring.cur)

    my_ring.extend(range(6,11))
    print(my_ring.data)
    print(my_ring.cur)

    my_ring.extend(range(11,115))
    print(my_ring.data)
    print(my_ring.cur)