#!/usr/bin/env python3

import os
import scipy.io.wavfile as wav

import numpy as np
from python_speech_features import mfcc
from features_utils_text import text_to_char_array, normalize_txt_file

def load_wavfile(wavfile):
	rate, sig = wav.read(wavfile)
	data_name = os.path.splitext(os.path.basename(wavfile))[0]
	return rate, sig, data_name

def get_audio_and_transcript(txt_files, wav_files, n_input, n_context):
	audio = []
	audio_len = []
	transcript = []
	transcript_len = []

	for txt_file, wav_file in zip(txt_files, wav_files):
		pass