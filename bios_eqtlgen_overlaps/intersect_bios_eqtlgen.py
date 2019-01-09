import pandas as pd
from optparse import OptionParser


def intersect_by_columns(
        left_file, right_file, left_index_col, right_index_col):
    """Intersect two dataframe by index specified by columns"""

    left_dataframe = pd.read_table(
        left_file, header=0, index_col=left_index_col)

    right_dataframe = pd.read_table(
        right_file, header=0, index_col=right_index_col)

    left_indexs = left_dataframe.index
    right_indexs = right_dataframe.index
    overlaps = [x for x in left_indexs if x in right_indexs]

    left_output_file = 'left_intersected.tsv'
    right_output_file = 'right_intersected.tsv'

    print('left input: {}; left output: {}'.format(left_file, left_output_file))
    print('right input: {}; right output: {}'.format(
        right_file, right_output_file))

    with open(left_output_file, 'w') as left_output_handler:
        left_overlap_dataframe = left_dataframe.loc[overlaps, :]
        left_overlap_dataframe.to_csv(left_output_handler, sep='\t')

    with open(right_output_file, 'w') as right_output_handler:
        right_overlap_dataframe = right_dataframe.loc[overlaps, :]
        right_overlap_dataframe.to_csv(right_output_handler, sep='\t')

    with open('left_rigth_overlaps.tsv', 'w') as overlaps_output:
        overlaps.to_frame().to_csv(overlaps, sep='\t')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-l', '--left-file', dest='left_file',
                      help='the name of left file')
    parser.add_option('-r', '--right-file', dest='right_file',
                      help='the name of right file')

    opts, args = parser.parse_args()
    left_file = opts.left_file
    right_file = opts.right_file

    bios_index_cols = (1, 2, 3, 4)
    eqtlgen_index_cols = (2, 3, 4, 5)
    intersect_by_columns(left_file, right_file,
                         bios_index_cols, eqtlgen_index_cols)
