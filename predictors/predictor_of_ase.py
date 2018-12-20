#!/path/to/python
# -*- coding: utf-8 -*-

# import os
import sys
import time
from os.path import join
from functools import wraps
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportWarning as w:
    print(w, file=sys.stderr)
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt


def timmer(func):
    """Print the runtime of the decorated function

    :params func(callable): function to be timmed
    :returns callable
    """

    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        fn = func.__name__
        rt = time.perf_counter() - start_time
        sys.stderr.write('{} is done; elapsed: {:.5f} secs\n'.format(fn, rt))
        return value

    return wrapper_timmer


class ASEPredictor:
    """Predictor of ASE
    """

    def __init__(self, ifn):
        """Initialization

        :params ifn(string): input file name
        """

        self.input_file_name = ifn
        self.raw_dataframe = None

    def debug(self):
        """Method to debug the class
        """
        pass

    def run(self):
        """Excute chosen methods
        """

    def get_input_file_name(self):
        """Return the name of input file
        """
        return self.input_file_name

    def load_file_into_dataframe(self):
        """Load input file
        """

        try:
            file_handler = open(self.input_file_name, 'r')
        except PermissionError(
            'Failed to read {}'.format(self.input_file_name)
                ) as e:
            print(e)
        else:
            with file_handler:
                self.raw_dataframe = pd.read_table(file_handler, header=0)

    def imputation(self):
        """Imputation missing values

        """
        pass

    def draw_figs(self):
        """Draw figures
        """
        fig, ax = plt.subplots()
        ax.plot()


def main():
    """Main function to run the module

    :params None
    :returns None
    :examples:

    >>> imoprt ASEPredictor
    >>> ipf = 'input.tsv'
    >>> ap = ASEPredictor(ipf)
    >>> ap.run()
    """

    FILE_PATH = [
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10', 
        'biosGavinOlCv10AntUfltCstLog2FCVar.tsv'
    ]

    input_file = join(FILE_PATH)
    ap = ASEPredictor(input_file)
    ap.debug()


if __name__ == '__main__':
    main()
