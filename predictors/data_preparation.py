#!/path/to/python3
# -*- coding: utf-8 -*-

import pysam
import sys
from FileReader import FileReader


input_file_name = '../utls/toy.sam'
vcf_file = FileReader(input_file_name)


class SamReader:
    """
    """

    def __init__(self, input_file_name):
        """
        """

        self.input_file_name = input_file_name
        try:
            self.sam_file = pysam.AlignmentFile(input_file_name, 'r')
        except IOError('Failed to read: {}'.format(input_file_name)) as e:
            raise e

    def read_file(self, method=None, mode=None, **kwargs):
        """Create file handler

        Args:
            method (string or None):
                method used to create file handler
            mode (string or None):
                mode used to load the file

        Returns:
            file handler
        """

        read_method_pool = [
            'plain',
            'sam',
            'bam',
            'fasta',
            'fastq',
            'vcf',
            'bcf'
        ]

        # default reading method is plain
        if method not in read_method_pool:
            method = 'plain'
            print(
                'Unknown method: {},'.format(method) + 'use plain as default',
                file=sys.stderr
                )
        elif method is None:
            method = 'plain'
        else:
            print(
                'Read file{0} as {1} file'.format(
                    self.input_file_name, method
                ),
                file=sys.stderr
            )

        mode_pool = [
            'r',
            'rb'
        ]

        if mode is None:
            mode = 'r'
        elif mode not in mode_pool:
            print('Unknown mode: {}, use r as default'.format(mode))
            mode = 'r'

        try:
            if method == 'plain':
                self.file_handler = open(self.input_file_name, mode)
            self.file_handler = pysam.AlignmentFile(self.input_file_name, mode)
        except PermissionError('Permission denied') as e:
            raise e
        except Exception('Unexpected error!!') as e:
            raise e
        else:
            return self.file_handler
