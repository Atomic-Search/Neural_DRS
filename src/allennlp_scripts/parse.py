#!/usr/bin/env python3
# -*- coding: utf8 -*-

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
import postprocess as pp


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
no_sep = "--no_sep"
MIN_TOKENS = 10

# NOTE:
# Use this for character-level exps
#no_sep = ""
#MIN_TOKENS = 20

# Settings for experiments
FORCE_PP = True       # whether we force postprocessing or skip if already there
SILENT = "--silent"   # silent parsing
FORCE = "-f"      # force reprocessing if directory already exists

# For tarring our own models  ??
VOCAB = "vocabulary/"
CONFIG = "config.json"
model_file = "model.tar.gz"
#/settings.sh

#parse.sh
# Parse raw text with an AllenNLP model
# Note: this script does not work when using multiple encoders
# Arguments: $1: input raw file (not tokenized!)
#        $2: model file
#        $3: target vocab

CURRENT_MODEL = f"{CWD}/models/allennlp/bert_char_1enc.tar.gz"
VOCAB_FILE = f"{CWD}/vocabs/allennlp/tgt_bert_char_1enc.txt"

os.environ["PYTHONPATH"] = DRS_GIT + "/evaluation/:${PYTHONPATH}"
os.environ["PYTHONPATH"] = NEURAL_GIT + "/src/:${PYTHONPATH}"

def parse(text, 
      model=CURRENT_MODEL, 
      vocab_file=VOCAB_FILE):
    """[summary]

    Args:
        text ([type]): [description]
        model ([type], optional): [description]. Defaults to CURRENT_MODEL.
        vocab_file ([type], optional): [description]. Defaults to VOCAB_FILE.
    """
    # Put raw text in format we read with our dataset reader, i.e. add dummy DRS after each line
    text = text.strip()
    text = text + "\tDummy"

#alp_file=${raw_file}.alp
#cp $raw_file $alp_file
#TAB=$'\t'
#sed -e "s/$/${TAB}Dummy/" -i $alp_file

    # Do the predicting -- to start we'll use the shell command. Pythonize this later.
    
    os.system("allennlp predict model text --use-dataset-reader --cuda-device 0 --predictor seq2seq $SILENT")

#out_file=${raw_file}.drs
#allennlp predict $cur_model $alp_file --use-dataset-reader --cuda-device 0 --predictor seq2seq --output-file $out_file $SILENT

    output = do_postprocess(args)

# Now do postprocessing, replace ill-formed DRSs by dummies
python $PP_PY --input_file $out_file 
        --output_file ${out_file}.out  
        --sig_file $SIG_FILE 
        --fix 
        --json 
        --sep $SEP 
        -rcl $REMOVE_CLAUSES 
        -m $MIN_TOKENS 
        -voc $vocab_file 
        $no_sep

# Remove temporary .alp file (clean up)
rm $alp_file


def main():
    """ """


if __name__ == "__main__":
    """  """
    main()
