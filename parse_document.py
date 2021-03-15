#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
This module takes a list of document file names (these should be in json),
breaks each one up into sentences, and feeds them into the DRS parser. It
outputs the resulting DRSs.
"""

from parse import Drs
import json
import os
import argparse
from spacy.lang.en import English
from mvp.single_article_search.AtomicCloud import bulkPushToElastic


def main(directory, max):
    """  """
    # Initialize spacy sentencizer.
    nlp = English()
    nlp.add_pipe("sentencizer")
    
    print(f"Getting list of files from directory: {directory}.")
    filenames = []
    drss = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filenames.append(filename)
    except:
        print(f"Can't find directory: {directory}")
    if len(filenames) < 1:
        raise Exception(f"No JSON files found in directory: {directory}")
    else:
        print(f"Files found: {len(filenames)}")
    all_drss = []
    for filename in filenames:
        doc_drss = []
        with open("path_to_file/person.json", "r") as f:
            json_doc = json.load(f)
        body = json_doc['text']
        doc = nlp(body)
        for sentence in doc.sents:
            parse_object = Drs(sentence.text)
            drs_list = parse_object.parsed_drs
            drs_dict = {'sentence': sentence}
            for box in drs_list:
                drs_dict[box['box_id']] = box
            all_drss.append(drs_dict)
    bulkPushToElastic(all_drss, "discourse_representation_structures", verbose=False)


if __name__ == "__main__":
    """  """
    max = 0
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("directory", 
                        help="The absolute path to the directory in which \
                            the input documents are found.")
    parser.add_argument("-m", "--max", action="store", dest="max",
                        help="Optionally enter a maximum number of \
                            documents to process.")
                 
    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    args = parser.parse_args()
    if (not isinstance(max, int) or max < 1):
        raise TypeError("Only positive integers allowed for optional \
                        --max argument.")
    
    main(args)
