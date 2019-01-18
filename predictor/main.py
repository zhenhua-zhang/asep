#!./env/bin/python
# -*- coding: utf-8 -*-
"""Main interface for asep"""

import os

from asep.utilities import save_obj_into_pickle
from asep.predictor import ASEPredictor

def main():
    """Main function to run the module """
    input_file = os.path.join(
        '/home', 'umcg-zzhang', 'Documents', 'projects', 'ASEpredictor',
        'outputs', 'biosGavinOverlapCov10',
        'biosGavinOlCv10AntUfltCstLog2FCBin.tsv'
    )
    asep = ASEPredictor(input_file)
    asep.run()
    save_obj_into_pickle(obj=asep, file_name="train_obj")


if __name__ == '__main__':
    main()
