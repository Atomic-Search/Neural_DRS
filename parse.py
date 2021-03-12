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

Output: A discourse representation structure object.
"""

import pprint
import os
import sys
from collections import defaultdict
import copy
from settings import (OUTPUT_FILE, INPUT_FILE, FINAL_FILE, PP_PY, 
                      CURRENT_MODEL, VOCAB_FILE, SILENT, SIG_FILE,
                      SEP, REMOVE_CLAUSES, MIN_TOKENS, NO_SEP)



class Drs:
    def __init__(self, text, model=CURRENT_MODEL, vocab=VOCAB_FILE):
        self.text = text
        self.model = model
        self.vocab = vocab
        self.drs = self.parse_text(text, model, vocab)
        self.parsed_drs = self.parse_drs(self.drs)
    
    def __repr__(self):
        return self.text + "\n\n" + pprint.pformat(self.parsed_drs)
    
    def parse_text(self, text, model, vocab):
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
        
    def parse_drs(self, drs):
        box_id = ""
        boxes = []
        box = defaultdict(list)
        drs = drs.strip()
        drs = drs.replace("\n\n", "\n")
        clauses = drs.split("\n")
        for clause in clauses:
            clause = clause.strip()
            terms = clause.split(" ")
            # import pdb; pdb.set_trace()
            if terms[0] == "b1":
                box["box_id"] = "b1"
                box_id = "b1"
            if box_id != terms[0] and terms[0] != "b1":
                box["box_id"] = box_id
                boxes.append(copy.deepcopy(box))
                box.clear()
                box_id = terms[0]
                box["box_id"] = box_id
            # All caps labels are always relations between boxes.
            # We will treat REFs different though.
            if terms[1].isupper() and terms[1] != "REF":
                box[terms[1]] = " ".join(terms[1:])
            elif terms[1] == "REF":
                box['REFS'].append(terms[-1])
            elif '"now"' in clause:
                tensed_var = terms[-2]
                tense = terms[1]
                box["tensed"] = str(tensed_var)
                box["tense"] = str(tense)
            #elif "time" in clause:
            #    terms = clause.split('"')
            #    variables = terms[-1].strip()
            #    box['roles'].append(clause)
            elif terms[1][0].isupper() and terms[1][1].islower() and terms[1] != "Name":
                # These are semantic roles.
                role = " ".join(terms[1:])
                box[terms[1]] = str(role)
                box["roles"].append(role)
            else:
                # The rest should be lexical items.
                lexical_item = " ".join(terms[1:])
                box[terms[1]] = lexical_item
                box["lexical_items"].append(lexical_item)
                if "v." in lexical_item:
                    box["verbs"].append(lexical_item)
                elif "n." in lexical_item:
                    box["nouns"].append(lexical_item)
                elif "a." in lexical_item:
                    box["mono"].append(lexical_item)
        boxes.append(copy.deepcopy(box))
        return boxes 
    
    def raw(self):
        print(self.drs)
                
    def pp_drs(self):
        parsed_drs = self.parsed_drs
        output = ""
        # "box relations" is any relation between boxes.
        # These are usually presupposition, negation, and
        # so on.
        box_relations = []
        last_box = ""
        for box in parsed_drs:
            # Output any relations that are pointing to this box.
            for box_relation in box_relations:
                if box_relation[-1] == box['box_id']:
                    output = output + f"  {box_relation[0]}\n"
                    box_relations.remove(box_relation)
            # Collect any additional relations from this box.
            keys = box.keys()
            for key in keys:
                if key.isupper():
                    terms = box[key].split(" ")
                    box_relations.append([key, terms[-1]])
            # Everything else belongs within the current box.
            if "REFS" in keys:    
                output = output +  "_____________________________\n"  #30 spaces
                output = output + f"| {' '.join(box['REFS'])}        {box['box_id']} |\n"
                output = output +  "_____________________________\n"
            else:
                output = output +  "_____________________________\n"  #30 spaces
                output = output + f"|              {box['box_id']} |\n"
                output = output +  "_____________________________\n"
            #if "pres" in keys:
            #    for pre in box['pres']:
            #        output = output + f"{pre}\n"
            if "lexical_items" in keys:
                for lexical_item in box['lexical_items']:
                    output = output + f"{lexical_item}\n"
                    variable = lexical_item.split(' ')[-1]
                    if "roles" in keys:
                        for role in box['roles']:
                            pointer = role.split(' ')[-2]
                            if variable == pointer:
                                box['roles'].remove(role)
                                output = output + f"    {role}\n"
            if len(box['roles']) > 0:
                for role in box['roles']:
                    output = output + f"    {role}\n"
            if "tense" in keys:
                output = output + f"    {box['tensed']} "
                if box['tense'] == "TPR":
                    output = output + f"< 'now'\n"
                elif box['tense'] == "EQU":
                    output = output + f"= 'now'\n"
            output = output + "_____________________________\n\n"
        print(output)                 
                    
                                
                    
                    
                    
            #         f"""
            #                 ______________________________
            #                 | {terms[-1]}\t\t\t\t|{box[box_id]}| |
            #                 ------------------------------
            #                 """)
                
            # elif :
                