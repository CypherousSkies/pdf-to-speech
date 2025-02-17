import os
import re
from difflib import SequenceMatcher
import nltk
import torch
from nltk.tag import pos_tag
from ocrmypdf import ocr
from spellchecker import SpellChecker
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pdftotext import PDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from reading4listeners import lang_dict


def string_metric(str1, str2):
    score = SequenceMatcher(None, str1, str2).ratio()
    return score


odds = {'‘': "'", '’': "'", '“': '"', '”': '"', '«': '"', '»': '"', '': '', '-\n': '', '|': '', '…': '', '_': ' ',
        '—': '', '...': '.', '" "': '', '  ': ' ', ".!": "!", "!.": "!", "?.": "?", ".?": "?", "?!": "?", "!?": "?",
        "{": "(", "}": ")", "#": "hashtag"}
odds = dict((re.escape(k), v) for k, v in odds.items())
odd_re = re.compile("|".join(odds.keys()))
spec = {'\n': ' ', '\\': ' ', '\"': ' " ', '-': ' ', '|': ' | ',
        '[': ' [ ', ']': ' ] ', ',': ' , ', '.': ' . ', '!': ' ! ',
        '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ', '—': ' ',
        '(': ' ( ', ')': ' ) ', "s'": "s '", ":": " : ", ";": " ; "}
spec = dict((re.escape(k), v) for k, v in spec.items())
spec_re = re.compile("|".join(spec.keys()))


class TextProcessor:
    def __init__(self, bert_model="distilbert-base-multilingual-cased", langs="en"):
        try:  # Use cached version if possible (making offline-mode default)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, local_files_only=True)
            self.model = AutoModelForMaskedLM.from_pretrained(bert_model, local_files_only=True)
        except:  # Models not cached
            print("> Downloading BERT models")
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, force_download=True)
            self.model = AutoModelForMaskedLM.from_pretrained(bert_model, force_download=True)
        self.sc = SpellChecker(distance=1, language=langs)
        self.langs = langs
        if langs is list:
            self.lang = [lang_dict[l][2] for l in langs]
        else:
            self.lang = lang_dict[langs][2]
        print("> BERT initialized")

    def loadepub(self, filename, sesspath, skip_correction=True):
        book = epub.read_epub(sesspath+filename)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        #chapters = [c for c in items if 'chapter' in c.get_name()]
        text0 = ""
        for c in items:#chapters:
            soup = BeautifulSoup(c.get_body_content(),'html.parser')
            text = soup.text.replace('\n', ' ').strip()
            text0 += text
        if skip_correction:
            return text0
        else:
            return self.correct_text(text0)

    # get and correct text
    def loadpdf(self, filename, sesspath, force=True, skip_correction=False, skip_ocr=False):
        if skip_ocr:
            with open(sesspath + filename, 'rb') as f:
                pdf = PDF(f)
                text0 = "\n\n".join(pdf)
        else:
            text0 = self._load(filename, sesspath, force)
            os.remove(sesspath + "tmp/tmp.pdf")
            os.remove(sesspath + "tmp/tmp.txt")
        if skip_correction:
            return text0
        else:
            return self.correct_text(text0)

    def correct_text(self, text0):
        text, text_original, incorrect = self._preprocess(text0)
        incorrect_ratio = len(incorrect) / len(text.split(" "))
        if incorrect_ratio > 0.1:  # if more than 10% of the words are wrong, it's possible there's another language mucking it up
            print(
                f"> {incorrect_ratio * 100}% of words marked wrong, this might be because the language was marked incorrectly.")
            from langdetect import detect
            lang = detect(text0)
            diff_lang = False
            if self.langs is list:
                diff_lang = lang not in self.langs
            else:
                diff_lang = lang is not self.langs
            if diff_lang:
                print(f"> Detected {lang} instead of initialized {self.langs}")
                self.sc = SpellChecker(distance=1, language=lang)
                self.langs = lang
                self.lang = lang_dict[lang][2]
                text, text_original, incorrect = self._preprocess(text0)
            else:
                print(f"> Lang detection resulted in no change, so probably just poor OCR. Continuing.")
        text = self._correct(text, text_original, incorrect)
        return text

    def _load(self, filename, sesspath, force):
        tpath = sesspath + "tmp/tmp.txt"
        if not os.path.isdir(sesspath + "tmp/"):
            os.mkdir(sesspath + "tmp/")
        ocr(sesspath + filename, sesspath + "tmp/tmp.pdf", sidecar=tpath, redo_ocr=(not force), deskew=force,
            rotate_pages=force, remove_background=force, clean=force, force_ocr=force, language=self.lang)
        with open(tpath, "r") as txt:
            text = txt.read()
        print("> OCR complete")
        return text

    # from Ravi Ilango's Medium Post
    def _preprocess(self, text):
        text = re.sub("\n\d+\n", "", text)
        # text = re.sub(page_numbers, '', text)
        text = odd_re.sub(lambda m: odds[re.escape(m.group(0))], text)
        text = re.sub(' " ', " ", text)
        rext = re.sub('".', " ", text)
        text = re.sub("\(\d+\)", "", text)
        text_original = text
        # cleanup text
        text = spec_re.sub(lambda m: spec[re.escape(m.group(0))], text)
        text = re.sub("\d+", " ", text)
        text = re.sub("'(?!s )", " ' ", text)

        def get_personslist(text):
            try:
                tagged = set(pos_tag(text.split()))
            except:
                nltk.download('averaged_perceptron_tagger')
                tagged = set(pos_tag(text.split()))
            return list(set([word for word, pos in tagged if pos == 'NNP']))

        personslist = get_personslist(text)
        # print(personslist)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'", "[", "]", "|", ":", ";"]
        # using pyspellchecker, identify incorrect words
        self.sc.word_frequency.load_words(ignorewords)
        self.sc.word_frequency.load_words(["pelargonium", "pelargoniums"])
        words = text.split()  # originally text.split()
        incorrectwords = [w for w in words if not list(self.sc.known([w])) and len(w) > 1]
        # replace incorrect words with [MASK]
        mask = self.tokenizer.mask_token
        for w in incorrectwords:
            text = text.replace(w + " ", mask, 1)
            # text_original = text_original.replace(w, mask)
        print(f"> {len(incorrectwords)} words to fix")
        print("> preprocessed text")
        return text, text_original, incorrectwords

    def _correct(self, text, text_original, incorrectwords):
        max_tokens = 512
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        inids = tokenized_text["input_ids"]
        attid = tokenized_text["attention_mask"]
        normins = list(torch.split(inids, max_tokens, dim=1))
        normats = list(torch.split(attid, max_tokens, dim=1))
        print("> running BERT")
        for inz, atz in tqdm(list(zip(normins, normats))):
            toktxt = {"input_ids": inz, "attention_mask": atz}
            mask_token_ids = torch.where(toktxt["input_ids"] == self.tokenizer.mask_token_id)[1].tolist()
            if mask_token_ids == []:
                continue
            with torch.no_grad():
                token_logits = self.model(**toktxt).logits
            for token_id, word in zip(mask_token_ids, incorrectwords):
                mask_token_logits = token_logits[0, token_id, :]
                top_tokens = torch.topk(mask_token_logits, 50).indices.tolist()
                choose = ""
                score = -1
                candidates = self.sc.candidates(word)
                for token in top_tokens:
                    unmask = self.tokenizer.decode(token)
                    mscore = string_metric(word, unmask)
                    for c in candidates:
                        mscore += string_metric(word, unmask)
                    if mscore > score and len(unmask) > 1:
                        choose = unmask
                if choose.count("##") > 0:
                    replaced = text_original.replace(" " + word, choose.replace('##', ''), 1)
                    if replaced != text_original:
                        text_original = replaced
                    else:
                        text_original = text_original.replace(word, choose.replace('##', ''), 1)
                else:
                    text_original = text_original.replace(word, choose, 1)
            s = slice(len(mask_token_ids), None)
            incorrectwords = incorrectwords[s]
        print("> finished BERT correction")
        text_original = text_original.replace("\n", " ")
        return text_original
