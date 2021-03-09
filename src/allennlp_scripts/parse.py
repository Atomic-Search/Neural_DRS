#!/usr/bin/env python3
"""
This should serve as an interface between the Atomic pipeline and the 
Discourse Representation Theory model for parsing. It replaces "parse.sh".

For now we are working at the sentence level, but this will probably change
later. 

Input: Raw text sentence (or potentially more, up to 50 characters for
        now.

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

# Folders
NEURAL_GIT=$(pwd)
DRS_GIT="${NEURAL_GIT}/DRS_parsing/"
SRC_PYTHON="${NEURAL_GIT}/src/"

# Files
PP_PY="${SRC_PYTHON}postprocess.py"
SIG_FILE="${DRS_GIT}evaluation/clf_signature.yaml"
COUNTER="${DRS_GIT}evaluation/counter.py"

# Setting for postprocessing
REMOVE_CLAUSES=75
SEP="|||"
# Use this for word-based exps
no_sep="--no_sep"
MIN_TOKENS=10

# NOTE:
# Use this for character-level exps
#no_sep=""
#MIN_TOKENS=20

# Settings for experiments
FORCE_PP=true       # whether we force postprocessing or skip if already there
SILENT="--silent"   # silent parsing
FORCE="-f"          # force reprocessing if directory already exists

# For tarring our own models
VOCAB="vocabulary/"
CONFIG="config.json"
model_file="model.tar.gz"

def main():
    """ """


if __name__ == "__main__":
    """  """
    main()
