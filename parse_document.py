#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
This module takes a list of document file names (these should be in json),
breaks each one up into sentences, and feeds them into the DRS parser. It
outputs the resulting DRSs.
"""

# Hacky fix so that I can import from AtomicCloud.py.
# TODO: Definitely fix this when things are integrated
# into packages. Since this is still experimental,
# I'm not going to try to do that stuff right now.

import sys
sys.path.append('/home/ubuntu/src/mvp/single_article_search')


from parse import Drs
import json
import os
import argparse
from spacy.lang.en import English
from AtomicCloud import bulkPushToElastic, indexTeardown


def main(args):
    """  """
    directory = args.directory
    max_files = args.max
    teardown = args.teardown
    index = "discourse_representation_structures"
    if teardown:
        if input(f"Tear down index '{index}'? (y/n)\n") == "y":
            if input("Are you sure? (y/n)\n") == "y":
                indexTeardown(index)
                print(f"Tore down index {index}. Continuing...")
            else:
                print("Teardown aborted! Existing...")
                sys.exit(0)
    # Initialize spacy sentencizer.
    nlp = English()
    sentencizer = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sentencizer)
    
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
    i = 0
    for filename in filenames:
        i += 1
        if i > max_files:
            break
        with open(f"{directory}/{filename}", "r") as f:
            json_doc = json.load(f)
        body = json_doc['text']
        title = json_doc['title']
        url = json_doc['source']
        doc = nlp(body)
        for span in doc.sents:
            sentence = span.text
            parse_object = Drs(sentence)
            drs_list = parse_object.parsed_drs
            drs_dict = {'sentence': sentence, 'title': title, 'url': url}
            for box in drs_list:
                drs_dict[box['box_id']] = box
            bulkPushToElastic([drs_dict], "discourse_representation_structures", verbose=False)


if __name__ == "__main__":
    """  """
    
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("directory", 
                        help="The absolute path to the directory in which \
                            the input documents are found.")
    parser.add_argument("-m", "--max", action="store", type=int,
                        help="Optionally enter a maximum number of \
                            documents to process.")
    parser.add_argument("-t", "--teardown", action="store_true", default=False)

     
    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    args = parser.parse_args()
    if args.max < 1:
        raise TypeError("Only positive integers allowed for optional \
                        --max argument.")
    
    main(args)
