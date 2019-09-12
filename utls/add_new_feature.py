#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add new features to given file of annotations by CADD v1.4

"""

from argparse import ArgumentParser

def get_args():
    """Get args and return an `ArgumentParser` object
    """
    parser = ArgumentParser()
    _group = parser.add_argument_group("Input")
    _group.add_argument("-a", "--annotation-file", dest="annotation_file", type=str, required=True, help="File including annotation from CADD. Required.")

    # _group = parser.add_argument_group("Config")

    _group = parser.add_argument_group("Output")
    _group.add_argument("-o", "--output-file", dest="output_file", type=str, default="output.tsv", help="Specify the output file")


def adder(args):
    """Core function for current script

    Arguments:
        args {NameSpace} -- An instance of Namespace object from argparse module. The instance includes options and values from CLI
    """


def main():
    """The main entry of current script
    """


if __name__ == "__main__":
    main()
