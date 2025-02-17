import os

import nltk
import numpy as np
import psutil
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager
#from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from tqdm import tqdm

from reading4listeners import lang_dict


def split_into_sentences(string):
    try:
        sentences = nltk.sent_tokenize(string)
    except:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(string)
    return sentences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
manager = ModelManager()


class Reader:
    def __init__(self, outpath, lang='en', tts_name=None, voc_name=None, decoder_mult=3, max_ram_percent=0.6):
        self.outpath = outpath
        self.decoder_mult = decoder_mult
        self.max_ram_percent = max_ram_percent
        model_name, vocoder_name, _ = lang_dict[lang]
        if tts_name is not None:
            model_name = tts_name
        if voc_name is not None:
            vocoder_name = voc_name
        #print(model_name, vocoder_name)
        self.model_path, self.config_path, _ = manager.download_model(model_name)
        self.vocoder_path, self.vocoder_config_path = None,None
        if vocoder_name is not None:
            self.vocoder_path, self.vocoder_config_path, _ = manager.download_model(vocoder_name)
        self._init_tts()
        print("> reader initialized")
        return
    
    def _init_tts(self):
        if self.vocoder_path is None:
            #self.synth = Synthesizer(model_path, config_path)
            self.tts = TTS(model_path=self.model_path,config_path=self.config_path).to(DEVICE)
        else:
            #self.synth = Synthesizer(model_path, config_path, vocoder_checkpoint=vocoder_path, vocoder_config=vocoder_config_path)
            self.tts = TTS(model_path=self.model_path,vocoder_path=self.vocoder_path,vocoder_config_path=self.vocoder_config_path,config_path=self.config_path).to(DEVICE)
        return

    def _write_to_file(self, wav, fname):
        wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav = wav.astype(np.int16)
        fout = self.outpath + fname + '.mp3'
        a = AudioSegment(
            wav.tobytes(),
            frame_rate=self.tts.synthesizer.output_sample_rate,
            sample_width=wav.dtype.itemsize,
            channels=1
        )
        a.export(fout, format="mp3")
        del a
        print(f"| > Wrote {fout}")
        return fout, len(wav) / self.tts.synthesizer.output_sample_rate

    def do_tts(self, text, fname, manual_swap=True):
        print(f"> Reading {fname}")
        sens = self.tts.synthesizer.split_into_sentences(text) #split_into_sentences(text)  # overrides TTS's uh, underwhelming, sentence splitter
        sens = [s for s in sens if len(s.split(' ')) >= 2]  # remove empty sentences
        wav = None
        mem_tot = psutil.virtual_memory().total
        print(f"> Have {mem_tot / (1024 * 1024)}MB of memory total")
        audio_time = 0
        splits = 0
        for sen in tqdm(sens):
            if sen == "":
                continue
            if manual_swap:
                mem_last = psutil.Process().memory_info().rss
            self.tts.synthesizer.tts_model.decoder.max_decoder_steps = len(sen) * self.decoder_mult  # override decoder steps
            sen = " ".join([s for s in self.tts.synthesizer.split_into_sentences(sen) if
                            len(s.split(" ")) >= 2])  # TTS crashes on null sentences. this fixes that i think
            data = self.tts.synthesizer.tts(sen)
            if sen == "" or sen == " ":
                continue
            if wav is None:
                wav = np.array(data)
            else:
                wav = np.append(wav, data)
            del data
            if manual_swap:
                mem_tot = psutil.virtual_memory().total
                mem_use = psutil.Process().memory_info().rss
                #print(f"| {100 * mem_use / mem_tot}% memory used")
                #print(f"| {100 * (mem_use - mem_last) / mem_last}% more than last it")
                # Is the current RAM usage too high by % or has RAM usage stopped (e.g. SWAP is being used)? write sound to file
                if mem_use / mem_tot > self.max_ram_percent or 1.01 * mem_use <= mem_last:
                    self._write_to_file(wav, fname + str(splits))
                    splits += 1
                    del wav
                    #del self.tts
                    #self._init_tts()
                    wav = None
        audio_time = 0
        file = ""
        if wav is not None and splits > 0:
            self._write_to_file(wav, fname + str(splits))
            splits += 1
            del wav
            wav = None
        del self.tts
        if splits > 0:
            audio = AudioSegment.silent()
            print(f"> Collecting {splits} files to final mp3")
            for i in tqdm(range(splits)):
                file = self.outpath + fname + f'{i}.mp3'
                audio += AudioSegment.from_mp3(file)
                os.remove(file)
            audio_time = len(audio) / 1000
            file = self.outpath + fname + '.mp3'
            audio.export(file, format='mp3')
        elif wav is not None and splits == 0:
            file, audio_time = self._write_to_file(wav, fname)
        else:
            raise Exception("Somehow reading4listeners.util.reader.wav is None")
        print(f"> Saved as {file}")
        self._init_tts()
        return file, audio_time
