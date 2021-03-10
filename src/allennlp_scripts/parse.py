#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
This should serve as an interface between the Atomic pipeline and the 
Discourse Representation Theory model for parsing. It replaces "parse.sh".

For now we are working at the sentence level, but this will probably change
later. 

Note: this script does not work when using multiple encoders.

Input: Raw text sentence (or potentially more, up to 50 characters for
    now).

Output: A discourse representation structure that analyzes the meaning
    of the input text.

TODO: Add switch for outputing human-readable DRSs for debugging, etc.

NB: Invoke this from the root directory like so:
python ./src/allennlp_scripts/parse.py
"""

__author__ = "Alan Hogue"
__version__ = "0.1.0"


import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()

#settings.sh
# Folders
NEURAL_GIT = CWD
DRS_GIT = NEURAL_GIT + "/DRS_parsing"
SRC_PYTHON = NEURAL_GIT + "/src"

# Files
PP_PY = SRC_PYTHON + "/postprocess.py"
SIG_FILE = DRS_GIT + "/evaluation/clf_signature.yaml"
COUNTER = DRS_GIT + "/evaluation/counter.py"

# Setting for postprocessing
REMOVE_CLAUSES = 75
SEP = "|||"
# Use this for word-based exps
NO_SEP = "--no_sep"
MIN_TOKENS = 10

# NOTE:
# Use this for character-level exps
#NO_SEP = ""
#MIN_TOKENS = 20

# Settings for experiments
FORCE_PP = True       # whether we force postprocessing or skip if already there
SILENT = "--silent"   # silent parsing
FORCE = "-f"      # force reprocessing if directory already exists

# For tarring our own models  ??
VOCAB = "vocabulary/"
CONFIG = "config.json"
MODEL_FILE = "model.tar.gz"
#/settings.sh

CURRENT_MODEL = f"{CWD}/models/allennlp/bert_char_1enc.tar.gz"
VOCAB_FILE = f"{CWD}/vocabs/allennlp/tgt_bert_char_1enc.txt"
INPUT_FILE = "drs_parse_input.drs"
OUTPUT_FILE = "drs_parse_output.txt"

os.environ["PYTHONPATH"] = DRS_GIT + "/evaluation/:${PYTHONPATH}"
os.environ["PYTHONPATH"] = NEURAL_GIT + "/src/:${PYTHONPATH}"

def parse(text, 
        model=CURRENT_MODEL, 
        vocab_file=VOCAB_FILE,
        for_humans=False,
        ):
    """
    This function is an interface around Neural DSR's prediction
    commands and assorted settings. It is intended to be imported
    into the Atomic pipeline, processing one sentence (or unit
    of text) at a time and returning the machine-readable discourse
    representation structure.

    Args:
        text        (string):           A plain string of one or more sentences.
        model       (str, optional):    Location of the model to use. Defaults to CURRENT_MODEL.
        vocab_file  (str, optional):    Location of the vocab to use. Defaults to VOCAB_FILE.
        for_humans  (bool, optional):   True if you want DRS formatted for human consumption.
    """
    # Put raw text in format we read with our dataset reader, i.e. add dummy DRS after each line
    text = text.strip()
    text = text + "\tDummy"

    # It's not very pretty, but for now we have to turn this input into a file. 
    # TODO: Figure out how to avoid these otherwise unnecessary files.
    with open(INPUT_FILE, "w") as inputf:
        print(text, file=inputf)

    # Do the predicting -- to start we'll use the shell command. Pythonize this later.
    os.system(f"allennlp predict {CURRENT_MODEL} {INPUT_FILE} --use-dataset-reader --cuda-device 0 --predictor seq2seq --output-file {OUTPUT_FILE} {SILENT}")

    # Now do postprocessing, replace ill-formed DRSs by dummies
    os.system(f"python {PP_PY} --input_file {OUTPUT_FILE} --output_file {OUTPUT_FILE}.out --sig_file {SIG_FILE} --fix --json --sep {SEP} -rcl {REMOVE_CLAUSES} -m {MIN_TOKENS} -voc {VOCAB_FILE} {NO_SEP}")

    # Read back in the resulting DRS.
    with open(f"{OUTPUT_FILE}.out", "r") as outputf:
        drs_parse = outputf.read()

    # Remove temporary files (clean up)
    os.system(f"rm {OUTPUT_FILE}; rm {INPUT_FILE}; rm {OUTPUT_FILE}.out")
    
    return drs_parse
