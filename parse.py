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
"""

import os
import sys
from settings import *


class Drs:
    def __init__(self, text, model=CURRENT_MODEL, vocab=VOCAB_FILE):
        self.text = text
        self.model = model
        self.vocab = vocab
        self.drs = self.parse(text, model, vocab)
    
    def parse(self, text, model, vocab_file):
        """
        This method is an interface around Neural DSR's prediction
        commands and assorted settings. It processes one sentence (or unit
        of text) at a time and returns the machine-readable discourse
        representation structure.

        Args:
            text        (string):           A plain string of one or more sentences.
            model       (str, optional):    Location of the model to use. Defaults to CURRENT_MODEL.
            vocab_file  (str, optional):    Location of the vocab to use. Defaults to VOCAB_FILE.
        """
        # Put raw text in format we read with our dataset reader, i.e. add dummy DRS after each line
        text = text.strip()
        text = text + "\tDummy"

        # It's not very pretty, but for now we have to turn this input into a file. 
        # TODO: Figure out how to avoid these otherwise unnecessary files.
        with open(INPUT_FILE, "w") as inputf:
            print(text, file=inputf)

        # Do the predicting -- to start we'll use the shell command. Pythonize this later.
        try:
            os.system(f"allennlp predict {CURRENT_MODEL} {INPUT_FILE} --use-dataset-reader --cuda-device 0 --predictor seq2seq --output-file {OUTPUT_FILE} {SILENT}")
        except:
            sys.stderr.write("Parsing command failed.")
        
        # Now do postprocessing, replace ill-formed DRSs by dummies
        try:
            os.system(f"python {PP_PY} --input_file {OUTPUT_FILE} --output_file {FINAL_FILE} --sig_file {SIG_FILE} --fix --json --sep {SEP} -rcl {REMOVE_CLAUSES} -m {MIN_TOKENS} -voc {VOCAB_FILE} {NO_SEP}")
        except:
            sys.stderr.write("Postprocessing command failed.")

        # Read back in the resulting DRS.
        with open(FINAL_FILE, "r") as outputf:
            drs_parse = outputf.read()

        # Remove temporary files (clean up)
        os.system(f"rm {OUTPUT_FILE}; rm {INPUT_FILE}; rm {FINAL_FILE}")
        
        return drs_parse

    def pretty_print(self, self.drs):
        print(self.drs)