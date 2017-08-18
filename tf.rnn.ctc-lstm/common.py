"""
    Definitions that don't fit elsewhere.
"""

import glob
import numpy as np
import cv2

# Constants
import time

SPACE_INDEX = 0
FIRST_INDEX = ord('0') - 1 # 0 is reserved to space

SPACE_TOKEN = '<space>'

__all__ = {
    "DIGITS",
    "sigmoid",
    "softmax",
}

OUTPUT_SHAPE = (64, 256)

DIGITS = "0123456789"
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CHARS = DIGITS
LENGTH = 16
LENGTHS = [16, 20] # the number of digits varies from LENGTH[0] to LENGTH[1] in an image.
TEST_SIZE = 200
ADD_BLANK = True # whether add a blank between digits
LEARNING_RATE_DECAY_FACTOR = 0.9
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000

# parameters for bilstm ctc
BATCH_SIZE = 64
BATCHES = 10

TRAIN_SIZE = BATCH_SIZE * BATCHES

MOMENTUM = 0.9

REPORT_STEPS = 100

# Hyper parameters
num_epochs = 200
num_hidden = 64
num_layers = 1

num_classes = len[DIGITS] +1 +1 # 10 integer + blank + ctc blank

def softmax(a):
    exps = np.exp(a.astype(np.float64))
    return exps / np.sum(exps, axis=-1)[:, np.newaxis]

def sigmoid(a):
    return 1./(1.+numpy.exp(-a))

"""
    {"dir_name":{"fname":(img, code)}}
"""
dataset = {}




