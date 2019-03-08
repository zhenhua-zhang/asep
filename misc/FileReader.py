from sys import stderr

# import pandas as pd
import pysam


class FileReader:
    """A class for file reading
    """

    def __init__(self, fn):
        """Initialization
        """
        self.file_name = fn

    def read_file(self, method=None, mode=None, **kwargs):
        """Read file by default method

        Args:
            method (None):
            mode (None):
            **kwargs:

        Returns:
        """

        method_pool = [
            'plain',   # As plain text
            'sam',     # As SAM with specific mode
            'todf'     # Read the input file into a Pandas DataFrame
            ]

        if method not in method_pool:
            raise ValueError(
                'Unknown method to read file: {} \n'.format(self.file_name)
                + 'Available method in cluding {}\n'.format(method_pool)
                )

        if method == 'plain':
            self.read_plain_file()
        elif method == 'sam':
            self.read_sam_file()
        elif method == 'todf':
            self.read_into_dataframe()

    def read_plain_file(self):
        """Read file as a plain text file
        """
        try:
            self.file_handler = open(self.file_name, 'r')
        except PermissionError('Permission denied!') as e:
            raise e
        except Exception('Unexpected exception') as e:
            raise e
        else:
            return self.file_handler

    def read_binary_file(self):
        """Read file as a binary file
        """
        pass

    def read_tar_file(self, mode=None):
        """Read tar file
        """

    def read_zip_file(self, mode='zip'):
        """Read file as a binary compressed file
        """

    def read_vcf_file(self, mode=None):
        """Read file as variant call format file
        """

    def read_sam_file(self, mode=None):
        """Read file as sequnce alignment map file
        """

        try:
            self.file_handler = pysam.AlignmentFile(self.file_name, mode)
        except PermissionError('Permission denied') as e:
            raise e
        except Exception('Unexpected error!!') as e:
            raise e
        else:
            return self.file_handler

    def read_into_dataframe(self, **kwargs):
        """Read the input file into a Pandas DataFame
        """


if __name__ == "__main__":
    stderr.write(
        "NOTE: please import it as module instead of executing it directly!\n"
        )
