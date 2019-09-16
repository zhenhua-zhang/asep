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
    _group.add_argument("-n", "--new-feature-file", dest="new_feature_file", type=str, required=True, help="The file including new features")
    _group.add_argument("-a", "--annotation-file", dest="annotation_file", type=str, required=True, help="File including annotation from CADD. Required.")

    _group = parser.add_argument_group("Config")
    _group.add_argument("-f", "--feature-column", dest="feature_column", type=int, default=5, help="Index(0-based) of the column of feature to be added. Default: 5")
    _group.add_argument("-N", "--feature-name", dest="feature_name", type=str, default=None, help="The name of newly added feature. Will inferer it if by default. Default: None")
    _group.add_argument("-m", "--missing-value", dest="missing_value", type=str, default="NA", help="Value used to fill the missing vlaue. Default: NA")
    _group.add_argument("-x", "--by-x-column", dest="by_x_column", nargs="+", type=int, default=[1, 2, 3, 4], help="Index(0-based) of column(s) by which annotations and new features are merged. Applied on current annotation file. Default: [1, 2, 3, 4]")
    _group.add_argument("-y", "--by-y-column", dest="by_y_column", nargs="+", type=int, default=[1, 2, 3, 4], help="Index(0-based) of column(s) by which annotations and new features are merged. Applied on new feature file. Default: [1, 2, 3, 4]")
    _group.add_argument("-p", "--insert-pos", dest="insert_pos", type=int, default=-1, help="Index(0-based) to insert the new feature. Append the new feature column if -1 (default). Default: -1")

    # TODO: New implementation to enable this flag
    # _group.add_argument("-s", "--sort-by-coord", dest="sort_by_coord", action="store_true", help="Whether sort the output file along the chromosome coordination.")

    _group = parser.add_argument_group("Output")
    _group.add_argument("-o", "--output-file", dest="output_file", type=str, default="output.tsv", help="Specify the output file")

    return parser

def get_value_by_index(input_list, idx):
    """Get specific value from a list (input_list) for given index supplied by idx"""
    return tuple([input_list[x] for x in idx])

def adder(args):
    """Core function for current script

    Arguments:
        args {Namespace} -- An instance of Namespace object from argparse module. The instance includes options and values from CLI
    """
    annotation_file = args.annotation_file
    new_feature_file = args.new_feature_file

    insert_pos = args.insert_pos
    by_x_column = args.by_x_column
    by_y_column = args.by_y_column
    feature_name = args.feature_name
    missing_value = args.missing_value
    feature_column = args.feature_column
    # sort_by_coord = args.sort_by_coord

    output_file = args.output_file

    new_feature_dict = {}
    with open(annotation_file, "r") as annotation_file_handle, \
        open(new_feature_file, "r") as new_feature_file_handle, \
        open(output_file, "w") as output_file_handle:

        feature_file_header = next(new_feature_file_handle, "EOF")
        if feature_name is None:
            feature_name = feature_file_header[feature_column]

        for _line in new_feature_file_handle:
            _line_list = _line.split("\t")  # A new CLI option to handle line delimiter
            _as_key = get_value_by_index(_line_list, by_y_column)
            new_feature_dict[_as_key] = _line_list[feature_column]

        annotation_file_header = next(annotation_file_handle, "EOF").split("\t")
        annotation_file_header.insert(insert_pos, feature_name)
        output_file_handle.write(annotation_file_header)

        for _line in annotation_file_handle:
            _line_list = _line.strip("\n").split("\t")  # A new CLI option to handle line delimiter
            _as_key = get_value_by_index(_line_list, by_x_column)
            _new_feature_value = new_feature_dict.get(_as_key, missing_value)
            _line_list.insert(insert_pos, _new_feature_value)
            output_file_handle.write("\t".join(_line_list) + "\n")


def main():
    """The main entry of current script"""
    parser = get_args()
    args = parser.parse_args()
    adder(args)


if __name__ == "__main__":
    main()
