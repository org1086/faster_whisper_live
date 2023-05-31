import logging
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word, Segment
from logger import initialize_logger
import numpy as np
from typing import NamedTuple, Iterable
import time

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
            # then trace back to update the last update data
            old_pos = self.cur
            self.cur = (self.cur + len(xs)) % self.max

            # update data to the ring
            if len(xs) >= self.max:
                update_data = xs[-self.max:]
                self.data[self.cur:] = update_data[:(self.max-self.cur)]
                self.data[:self.cur] = update_data[(self.max-self.cur):]
            else:
                if old_pos + len(xs) <= self.max:
                    self.data[self.cur:self.cur + len(xs) - 1] = xs
                else:
                    self.data[old_pos:] = xs[:self.max - old_pos]
                    self.data[:old_pos + len(xs) - self.max] = xs[self.max - old_pos:]

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
        if self.cur + len(xs) <= self.max:
            self.data.extend(xs)
            if len(self.data) == self.max:
                self.cur = 0
                self.__class__ = self.__Full
            else:
                self.cur += len(xs)
        else:
            # TRICK: add 0s to fill the buffer
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

class ShiftedWindow(NamedTuple):
    start: int                      # buffer start position
    length: int                     # buffer length
    end: int                        # buffer end position
    aligned_words: Iterable[Word]   # words with alignment

class LiveWhisperProcessorBase:

    def __init__(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")    

    def load_model(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")

    def transribe(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def align_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")

    def process(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def estimate_window_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def estimate_overlaping_window_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")    
    
    def process_confirmed_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def process_overlaping_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def process_new_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    

class LiveWhisperProcessor(LiveWhisperProcessorBase):
    def __init__(self, 
                 model_name:str = 'tiny', 
                 language: str = 'vi', 
                 device: str = 'cpu', 
                 compute_type: str = 'auto',
                 download_root: str = 'models/', 
                 local_files_only: bool = True,
                 window_samples: int = 160000,
                 sample_rate: int = 16000,
                 left_error_thresh: float = 0.15,
                 right_error_thresh: float = 0.2   
                ):
        
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.left_error_thresh = left_error_thresh
        self.right_error_thresh = right_error_thresh

        # ring buffer to store buffers to process
        self.audio_buffer = RingBuffer(self.window_samples)

        self.counter = 0                    # sample counter current sampling time

        self.previous_window = ShiftedWindow(0,0,0)
        self.current_window = ShiftedWindow(0,0,0)

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
            self.current_window.start = 0
            self.current_window.length = self.counter
            self.current_window.end = self.current_window.start + self.current_window.length
        else:
            self.current_window.start = self.counter - self.window_samples
            self.current_window.length = self.window_samples
            self.current_window.end = self.current_window.start + self.current_window.length
        
        segments, _ = self.model.transcribe(new_audio, language = self.language)

        aligned_words = self.align_words(segments, self.current_window.start)
        self.current_window.aligned_words = aligned_words

        return self.process()

    def process(self) -> TranscriptionResult:        
        """
        Process and return transcription result. Calculate and produce confirmed words,
        validating words by estimating overlapping windows over time. Required data:
        - current_window of type `ShiftedWindow`
        - previous_windos of type `ShiftedWindow`
        - Return: `TranscriptionResult` with last confirmed text and validating text.
        """


        pass

    def estimate_window_words(self, 
                              start: int, 
                              end: int, 
                              aligned_words: Iterable[Word]
                              ) -> Iterable[Word]:
        pass
    
    def estimate_overlaping_window_words(self, 
                              start: int, 
                              end: int
                              ) -> Iterable[Word]:
        pass

    def process_confirmed_words(self):
        if not len(self.previous_window.aligned_words):
            return None
        
        if self.previous_window.start < self.current_window.start:
            return self.estimate_window_words(self.previous_window.start, 
                                              self.current_window.start -1, 
                                              self.previous_window.aligned_words
                                              )       
        return None

    def process_overlaping_words(self):
        if not (len(self.previous_window.aligned_words) or len(self.current_window.aligned_words)) :
            return None
        
        if self.previous_window.end > self.current_window.start:
            return self.estimate_overlaping_window_words(self.current_window.start, 
                                                         self.previous_window.end)
        
        return None

    def process_new_words(self):
        if not len(self.current_window.aligned_words):
            return None
        
        if self.previous_window.end < self.current_window.end:
            return self.estimate_window_words(self.previous_window.end +1,                
                                              self.current_window.end,                
                                              self.current_window.aligned_words
                )
        
        return None
    
    def shift_word(self, word: Word, shifted_secs: int):
        word.start += shifted_secs
        word.end += shifted_secs
        return word
    
    def align_words(self, segments: Iterable[Segment], start_position: int):
        shifted_secs = start_position / self.sample_rate
        return [self.shift_word(word, shifted_secs) for segment in segments for word in segment]

if __name__ == "__main__":
    my_ring = RingBuffer(160000)

    start = time.time()

    # print(my_ring.data)
    # print(my_ring.cur)

    # my_ring.append(1)
    # print(my_ring.data)
    # print(my_ring.cur)

    # my_ring.extend(range(2,5))
    # print(my_ring.data)
    # print(my_ring.cur)

    # my_ring.append(5)
    # print(my_ring.data)
    # print(my_ring.cur)

    # my_ring.extend(range(6,11))
    # print(my_ring.data)
    # print(my_ring.cur)

    # my_ring.extend(range(11,115))
    # print(my_ring.data)
    # print(my_ring.cur)

    my_ring.extend(range(115,16000*1000))
    # print(my_ring.data)
    # print(my_ring.cur)

    end = time.time()

    print(f'total time execution: {end - start} secs.')