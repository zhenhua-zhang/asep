#!/path/to/python
# -*- coding: utf-8 -*-

# import os
# import sys
import time
from functools import wraps
from sys import stderr
# from utls import FileReader

# import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportWarning as w:
    print(w)
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt


def timmer(func):
    """Print the runtime of the decorated function
    """

    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        fn = func.__name__
        rt = time.perf_counter() - start_time
        #  st = time.perf_counter()
        stderr.write('{} is done; elapsed: {:.5f} secs\n'.format(fn, rt))
        return value

    return wrapper_timmer


class Log2fcPredictor:
    """Predictor of log2FC
    """

    def __init__(self, ifn):
        """Initialization
        """
        self.input_file_name = ifn

    def debug(self):
        """Method to debug the class
        """
        pass

    def run(self):
        """Excute chosen methods
        """

    def read_input_file(self):
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
                pd.read_table(self.input_file_name, header=0)

    def get_input_file_name(self):
        """Return the name of input file
        """
        return self.input_file_name

    def draw_figs(self):
        """Draw figures
        """
        fig, ax = plt.subplots()
        ax.plot()
