from datetime import datetime

MAX_CAP_VALUE = 600
MIN_CAP_VALUE = 0

capacitive_screen_shape = (6, 6)  # capacitive sensor resolution

z_threshold = -8

NAME = "None"
BODYPART = "None"
DATE = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
DIR = 'data/'
FILE_INDEX = 0
TAG = {
    0: "IDLE",
    1: "FINGER_PAD",
    2: "PALMAR_SIDE_FINGERS",
    3: "PALM",
    4: "HAND_ROCK_DOWN",
    5: "HAND_ROCK_UP",
    6: "MOUTH",
    7: "EAR",
}

LABEL = {
    1: "PRESS",
    2: "SHEAR_1LV",
    3: "SHEAR_2LV",
    4: "SHEAR_3LV",
    5: "FREE",
    0: "DISCARD"
}

SLIDING_WINDOW_SEC = 5