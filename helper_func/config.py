import os

ORIG_BASE_PATH = "matroot"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])


BASE_PATH = 'dataset'
POSITIVE_PATH = os.path.sep.join([BASE_PATH, "table"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_table"])


MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (224, 224)

MODEL_PATH = "table_detector3.h5"
ENCODER_PATH = "label_encoder_table2.pickle"

MIN_PROBA = 0.9
