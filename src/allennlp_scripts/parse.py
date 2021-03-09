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
"""

__author__ = "Alan Hogue"
__version__ = "0.1.0"


import os


FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    """ """


if __name__ == "__main__":
    """  """
    main()
