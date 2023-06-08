import logging
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word, Segment
from logger import initialize_logger, initialize_file_logger
import numpy as np
from typing import Iterable, List
import time
from override_ring import RingBuffer

logger = initialize_logger(__name__, logging.DEBUG)
file_logger = initialize_file_logger(__name__, logging.DEBUG)

class MutableWord:
    def __init__(self, start: float, end: float, word: str, probability: float):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability
    
    global to_mutable
    def to_mutable(word: Word):
        return MutableWord(word.start, 
                           word.end, 
                           word.word, 
                           word.probability
                           )
    def clone(self):
        return MutableWord(self.start, 
                           self.end, 
                           self.word, 
                           self.probability)
    
    def __str__(self) -> str:
        return f'{{start: {self.start}, end: {self.end}, word: "{self.word}", probability: {self.probability}}}'

class TranscriptionResult:
    def __init__(self, last_confirmed: str, validating: str):
        self.last_confirmed = last_confirmed
        self.validating = validating
        
    def __str__(self) -> str:
        return f'{self.last_confirmed} >>> {self.validating}'

class ShiftedWindow:
    def __init__(self, start: int =0, length: int =0, end: int =0, aligned_words: List[MutableWord]=[]):
        self.start = start                      # buffer start position
        self.length = length                    # buffer length
        self.end = end                          # buffer end position
        self.aligned_words = aligned_words      # words with alignment
    
    def clone(self):
        return ShiftedWindow(self.start, 
                             self.length, 
                             self.end, 
                             [w.clone() for w in self.aligned_words]
                             )
    def __str__(self) -> str:
        return f'{{start: {self.start}, length: {self.length}, end: {self.end}, aligned_words: {[str(w) for w in self.aligned_words]}}}'
    

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
    
    def process_confirmed_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def process_overlaping_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")    

    def get_window_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def process_new_words(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")
    
    def format_timestamped_words_as_str(self, **kwargs):
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
                 intersection_thresh: float = 0.7  
                ):
        
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.local_files_only = local_files_only
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.intersection_thresh = intersection_thresh      # if the intersect ratio of the extreme word with 
                                                            # the concerning window larger than this threshold
                                                            # then put this word into your bag.

        # ring buffer to store buffers to process
        self.audio_buffer = RingBuffer(self.window_samples)

        self.counter = 0                    # sample counter current sampling time

        self.previous_window = ShiftedWindow()
        self.current_window = ShiftedWindow()

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

    def transcribe(self, audio)-> TranscriptionResult:
        '''
        Transcribe audio frames using Whisper model.
        - audio: tranformed streaming audio buffer of type of float32-type array.
        - Return: `TranscriptionResult` with last confirmed text and validating text.
        '''
        # assign previous window as current window of the previous execution
        self.previous_window = self.current_window.clone()

        new_audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0

        # temp log for investigation
        file_logger.info('==========================================================')
        file_logger.info(f">>>>> new audio size: {len(new_audio)} samples of float type.")

        # push to the overridable buffer of the processor
        self.audio_buffer.extend(new_audio)
        self.counter += len(new_audio)
        audio_from_ring = np.array(self.audio_buffer.get())

        # BEGIN TEST `override ring buffer` correctness
        # logger.info(f'new audio length: {len(new_audio)}')
        # logger.info(f'ring buffer data length: {len(self.audio_buffer.data)}')
        # logger.info(f'ring buffer get() data length: {len(self.audio_buffer.get())}')
        # return TranscriptionResult('','')
        # END TEST

        # logger.info(f'audio_from_ring: {audio_from_ring}')
    
        if self.counter <= self.window_samples:
            self.current_window = ShiftedWindow(start = 0, 
                                                length = self.counter, 
                                                end = self.counter - 1
                                                )
        else:
            self.current_window = ShiftedWindow(start = self.counter - self.window_samples, 
                                                length = self.window_samples, 
                                                end = self.counter - 1
                                                )
        start = time.time()
        segments, _ = self.model.transcribe(audio_from_ring, 
                                            language = self.language,
                                            word_timestamps=True
                                            )
        segments = list(segments)       # convert segment iterator to list
        end = time.time()

        logger.info(f'-------------------------------------------------------')
        logger.info(f'>> buffer size from ring: {len(audio_from_ring)}')
        logger.info(f'>> actual inference time: {end-start}')

        # logger.info(f'segments: {segments}')

        start = time.time()
        aligned_words = self.align_words(segments, self.current_window.start)
        end = time.time()

        logger.info(f'>> words alignment time: {end-start}')


        # logger.info(f'aligned_words: {[str(w) for w in aligned_words]}')
        start = time.time()
        self.current_window.aligned_words = aligned_words
        # logger.info(f'current_window: {self.current_window}')
        end = time.time()

        logger.info(f'>> `aligned_words to current window` time: {end-start}')

        # temp log for investigation
        file_logger.info(f">>>>> previous window timestamp range (sec): [{self.previous_window.start/self.sample_rate:.2f} - {self.previous_window.end/self.sample_rate:.2f}]")
        file_logger.info(f">>>>> current window timestamp range (sec): [{self.current_window.start/self.sample_rate:.2f} - {self.current_window.end/self.sample_rate:.2f}]")

        # logger.info(f'previous window aligned words: {[str(w) for w in self.previous_window.aligned_words]}')
        # logger.info(f'current window aligned words: {[str(w) for w in self.current_window.aligned_words]}')

        return self.process()

    def process(self) -> TranscriptionResult:        
        """
        Process and return transcription result. Calculate and produce confirmed words,
        validating words by estimating overlapping windows over time. Required data:
        - current_window of type `ShiftedWindow`
        - previous_windos of type `ShiftedWindow`
        - Return: `TranscriptionResult` with last confirmed text and validating text.
        """

        start = time.time()

        # logging previous and current windows with timestamped words for investigation
        file_logger.info(">>>>> previous window's timestamped words:")
        file_logger.info(self.format_timestamped_words_as_str(self.previous_window.aligned_words))
        file_logger.info(">>>>> current window's timestamped words:")
        file_logger.info(self.format_timestamped_words_as_str(self.current_window.aligned_words))

        confirmed_words = self.process_confirmed_words()
        overlaping_words = self.process_overlaping_words()
        new_words = self.process_new_words()

        file_logger.info('>>>>> confirmed words:')
        file_logger.info(self.format_timestamped_words_as_str(confirmed_words))
        file_logger.info('>>>>> overlaping words:')
        file_logger.info(self.format_timestamped_words_as_str(overlaping_words))
        file_logger.info('>>>>> new words:')
        file_logger.info(self.format_timestamped_words_as_str(new_words))

        # logger.info(f'confirmed_words: {confirmed_words}')
        # logger.info(f'overlaping_words: {overlaping_words}')
        # logger.info(f'new_words: {new_words}')

        self.confirmed_words.extend(confirmed_words)
        self.validating_words = overlaping_words
        self.validating_words.extend(new_words)

        end = time.time()

        logger.info(f'>> processing (+confirmed,overlap,newwords) time: {end - start}')

        # logger.info(f'=> Full text: {"".join([w.word for w in self.confirmed_words])} >>> {"".join([w.word for w in self.validating_words])}')

        return TranscriptionResult(last_confirmed = ''.join([w.word for w in confirmed_words]),
                                     validating = ''.join([w.word for w in self.validating_words]))

    def process_confirmed_words(self) -> List[MutableWord]:
        confirmed_words = []

        # logger.info(f'process_confirmed_words -> previous_window: {self.previous_window}')
        # logger.info(f'process_confirmed_words -> end_sec: {(self.current_window.start - 1) / self.sample_rate}')

        if len(self.previous_window.aligned_words) and self.previous_window.start < self.current_window.start:            
            end_sec = (self.current_window.start - 1) / self.sample_rate

            # temp log for investigation
            file_logger.info(f">>>>> confirmed window timestamp range (sec): [{self.previous_window.start/self.sample_rate:.2f} - {end_sec:.2f}]")

            # logger.info(f'process_confirmed_words.end_sec: {end_sec}')
            
            for w in self.previous_window.aligned_words:
                if w.end <= end_sec:
                    confirmed_words.append(w)
                elif w.start < end_sec:
                    if (end_sec - w.start) / (w.end - w.start) > self.intersection_thresh:
                        confirmed_words.append(w)
                else:
                    break
        return confirmed_words

    def get_window_words(self, start_sec: int, end_sec: int, aligned_words: Iterable[MutableWord]) -> List[MutableWord]:
        window_words = []
        for w in aligned_words:
            if w.start >= start_sec:
                if w.end <= end_sec:
                    window_words.append(w)
                elif w.start < end_sec:
                    if (end_sec - w.start) / (w.end - w.start) > self.intersection_thresh:
                        window_words.append(w)
            elif w.end > start_sec:
                if (w.end - start_sec) / (w.end - w.start) > self.intersection_thresh:
                    window_words.append(w)
        return window_words
    
    def process_overlaping_words(self) -> List[MutableWord]:
        overlap_words = []
        
        if self.previous_window.end > self.current_window.start:
            start_sec = self.current_window.start / self.sample_rate
            end_sec = self.previous_window.end / self.sample_rate
            overlap_prev_words = self.get_window_words(start_sec, end_sec, self.previous_window.aligned_words)
            overlap_cur_words = self.get_window_words(start_sec, end_sec, self.current_window.aligned_words)

            # temp log for investigation
            file_logger.info(f">>>>> overlapping window timestamp range (sec): [{start_sec:.2f} - {end_sec:.2f}]")

            if not len(overlap_prev_words):
                # nothing to update to current overlaping window, just keep the aligned words as it is.
                overlap_words = overlap_cur_words
            elif not len(overlap_cur_words):
                # prepend overlap_prev_words to current window aligned words
                # since there're no words within current overlaping window.
                self.current_window.aligned_words = overlap_prev_words + self.current_window.aligned_words
                overlap_words = overlap_prev_words
            else:
                prev_words_confidence = sum([w.probability for w in overlap_prev_words])/ len(overlap_prev_words)
                cur_words_confidence = sum([w.probability for w in overlap_cur_words])/ len(overlap_cur_words)
                
                if prev_words_confidence > cur_words_confidence:
                    # prepend overlap_prev_words to current window aligned words
                    # since there're no words within current overlaping window.
                    self.current_window.aligned_words = overlap_prev_words + self.current_window.aligned_words
                    overlap_words = overlap_prev_words
                else:
                    # nothing to update to current overlaping window, just keep the aligned words as it is.
                    overlap_words = overlap_cur_words 
        return overlap_words

    def process_new_words(self) -> List[MutableWord]:
        new_words = []
        
        if len(self.current_window.aligned_words) and self.previous_window.end < self.current_window.end:
            start_sec = (self.previous_window.end + 1) / self.sample_rate

            # temp log for investigation
            file_logger.info(f">>>>> new window timestamp range (sec): [{start_sec:.2f} - {self.current_window.end/self.sample_rate:.2f}]")

            new_words = []
            for w in reversed(self.current_window.aligned_words):
                if w.start >= start_sec:
                    new_words.append(w)
                elif w.end > start_sec:
                    if (w.end - start_sec) / (w.end - w.start) > self.intersection_thresh:
                        new_words.append(w)
                else:
                    break

        return list(reversed(new_words))
    
    def shift_word(self, word: MutableWord, shifted_secs: int) -> MutableWord:
        word.start += shifted_secs
        word.end += shifted_secs
        return word
    
    def align_words(self, segments: Iterable[Segment], start_position: int) -> List[MutableWord]:
        shifted_words = []

        word_count = 0
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    word_count +=1
        logger.info(f'>> word count: {word_count}')

        shifted_secs = start_position / self.sample_rate     
        for segment in segments:
            if segment.words:
                shifted_words.extend([self.shift_word(to_mutable(word), shifted_secs) for word in segment.words])
        return shifted_words
    
    def format_timestamped_words_as_str(self, words: List[MutableWord]):
        printed_str = ''
        for w in words:
            printed_str += f'[{w.start:.2f} {w.word.strip()} ({100*w.probability:.1f}%) {w.end:.2f}] '
        return printed_str.rstrip()

if __name__ == "__main__":
    my_ring = RingBuffer(160000)

    my_window = ShiftedWindow(0,0,0,[])

    print(my_window)