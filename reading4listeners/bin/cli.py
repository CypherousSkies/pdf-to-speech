#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import os
import re
import sys
import time
from datetime import timedelta
from reading4listeners import lang_dict
from reading4listeners.util.reader import Reader
from reading4listeners.util.text import TextProcessor
from reading4listeners.util.ocr import TrOCR

os.environ["TOKENIZERS_PARALLELISM"] = "False"
tag_remover = re.compile('<.*?>')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_ext(filename):
    return filename.split(".")[-1]


def get_texts(sesspath, lang, skip_correction, skip_ocr, use_TrOCR):
    setup_time = time.time()
    tp = TextProcessor(langs=lang)
    if use_TrOCR:
        trocr = TrOCR()
    setup_time = time.time()-setup_time
    files = [f for f in os.listdir(sesspath) if get_ext(f) in ['pdf', 'txt', 'muse','epub']]
    run_times = {}
    word_counts = {}
    texts = [[] for _ in files]
    print(f"> Reading {files}")
    for i, filename in enumerate(files):
        print(f"> Loading {filename}")
        start = time.time()
        if get_ext(filename) == 'pdf':
            if use_TrOCR:
                text = trocr.extract_text(sesspath+filename)
            else:
                text = tp.loadpdf(filename, sesspath, force=True, skip_correction=skip_correction, skip_ocr=skip_ocr)
        elif get_ext(filename) == 'epub':
            text = tp.loadepub(filename, sesspath, skip_correction=skip_correction)
        elif get_ext(filename) == 'txt':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            if not skip_correction:
                text = tp.correct_text(text)
        elif get_ext(filename) == 'muse':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            text = re.sub(tag_remover, '', text)
            if not skip_correction:
                text = tp.correct_text(text)
        else:
            continue
        run_times[filename] = time.time()-start+setup_time
        word_counts[filename] = len(text.split(" "))
        texts[i] = text
    del tp
    if use_TrOCR:
        del TrOCR
    print("> Done text preprocessing")
    return texts, files, word_counts, run_times

def read_texts(texts, files, outpath, lang, max_mem, decoder_mult):
    setup_time = time.time()
    reader = Reader(outpath, lang=lang, decoder_mult=decoder_mult, max_ram_percent=max_mem)
    setup_time = time.time()-setup_time
    run_times = {}
    audio_times = {}
    for text, name in zip(texts, files):
        start = time.time()
        _, t = reader.do_tts(text, name)
        run_times[name] = time.time()-start+setup_time
        audio_times[name] = t
    del reader
    return audio_times, run_times


def main():
    parser = argparse.ArgumentParser(
        description="""Read PDFs into MP3 files!\n"""
                    """In the interests of user-friendliness, this cli will be kept pretty bare-bones"""
                    """
        Basic usage:
        
        $ reading4listeners [--in_path in/] [--out_path out/] [--lang en]
        
        Converts pdfs, txts, muses in the folder "in/" and output mp3s to the folder "out/" with the primary language set to "en"
        List languages:
        
        $ reading4listeners --list_languages
        
        Lists available languages (Warning! Not tested on non-latin scripts!)
            """
    )
    parser.add_argument("--in_path", type=str, default="in/", help="Path containing files to be converted.")
    parser.add_argument("--out_path", type=str, default="out/", help="Output path.")
    parser.add_argument("--lang", type=str, default="en", help="Two-letter language code.")
    parser.add_argument("--max_mem", type=float, default=60, help="Upper bound of memory usage (as percent of RAM)")
    parser.add_argument("--decoder_mult", type=int, default=3, help="Sets number of times the max decoder steps is "
                                                                    "relative to the length of a sentence (set to a "
                                                                    "smaller number if you get slurred speech, "
                                                                    "increase if sentences are getting cut off)")
    parser.add_argument(
        "--list_langs",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="List available languages.",
    )
    parser.add_argument(
        "--collect_time_data",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Write time taken data to `time_data.csv` for analysis",
    )
    parser.add_argument(
        "--skip_correction",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Skip using BERT to improve OCR results",
    )

    parser.add_argument(
        "--skip_ocr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use existing text layer instead of redoing OCR (PDF/DJVU only)",
    )

    parser.add_argument(
        "--use_TrOCR",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use TrOCR instead of tesseract[+BERT] (WIP)",
    )

    args = parser.parse_args()
    if args.list_langs:
        print(list(lang_dict.keys()))
        sys.exit()
    if not os.path.isdir(args.in_path):
        print("input path must exist and contain files!")
        parser.parse_args(["-h"])
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    run(args.in_path, args.out_path, args.lang, args.collect_time_data, args.max_mem, args.decoder_mult, args.skip_correction, args.skip_ocr,args.use_TrOCR)
    return


def run(in_path, out_path, lang, time_data, max_mem, decoder_mult, skip_correction, skip_ocr, use_TrOCR):
    start_time = time.time()
    texts, files, word_counts, run_t_times = get_texts(in_path, lang, skip_correction, skip_ocr, use_TrOCR)
    audio_times, run_r_times = read_texts(texts, files, out_path, lang, max_mem, decoder_mult)
    time_taken = time.time() - start_time
    if time_data:
        with open('time_data.csv', 'a') as f:
            writer = csv.writer(f)
            for name in files:
                writer.writerow([get_ext(name),word_counts[name], run_t_times[name]+run_r_times[name], audio_times[name]])
    print(f"> Read {sum(word_counts.values())} words in {timedelta(seconds=time_taken)} seconds with a real time factor of {time_taken / sum(audio_times.values())}")
    return


if __name__ == "__main__":
    main()
