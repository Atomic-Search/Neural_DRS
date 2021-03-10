#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
    This is a pythonized version of the original 
    allennlp_scripts/settings.sh
"""

#settings.sh
# Folders
NEURAL_GIT = os.path.normpath(os.path.dirname( __file__ ))
DRS_GIT = NEURAL_GIT + "/DRS_parsing"
SRC_PYTHON = NEURAL_GIT + "/src"

# Files
PP_PY = SRC_PYTHON + "/postprocess.py"
SIG_FILE = DRS_GIT + "/evaluation/clf_signature.yaml"
COUNTER = DRS_GIT + "/evaluation/counter.py"
FINAL_FILE = NEURAL_GIT + "/drs_representation.txt"

# Setting for postprocessing
REMOVE_CLAUSES = 75
SEP = '"|||"'
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

CURRENT_MODEL = f"{NEURAL_GIT}/models/allennlp/bert_char_1enc.tar.gz"
VOCAB_FILE = f"{NEURAL_GIT}/vocabs/allennlp/tgt_bert_char_1enc.txt"
INPUT_FILE = f"{NEURAL_GIT}/drs_parse_input.drs"
OUTPUT_FILE = f"{NEURAL_GIT}/drs_parse_output.txt"

os.environ["PYTHONPATH"] = DRS_GIT + "/evaluation/:${PYTHONPATH}"
os.environ["PYTHONPATH"] = NEURAL_GIT + "/src/:${PYTHONPATH}"