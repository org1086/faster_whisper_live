import soundfile as sf
import torchaudio
import kenlm # pip3 install https://github.com/kpu/kenlm/archive/master.zip
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

def get_transcription(processor, model, ngram_lm_model, audio_path='audio.wav'):
    
    speech, sr = torchaudio.load(audio_path)
    speech = speech.squeeze()

    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)

    input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
    logits = model(input_values).logits[0]
    # print(logits.shape)
    # decode ctc output
    #pred_ids = torch.argmax(logits, dim=-1)
    # greedy_search_output = processor.decode(pred_ids)
    beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
    # print("Greedy search output: {}".format(greedy_search_output))
    #print("Beam search output: {}".format(beam_search_output))
    return beam_search_output

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""

    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,language_model=LanguageModel(lm_model))
    return decoder
