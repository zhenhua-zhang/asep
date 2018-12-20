import pandas as pd

bios_input_file = 'bios_chr22.tsv'
bios_index_cols = (1, 2, 3, 4)
bios_dataframe = pd.read_table(
    bios_input_file, header=0, index_col=bios_index_cols
)

bdf_rows = bios_dataframe.index

eqtlgen_input_file = 'eqtlgen_chr22.tsv'
eqtlgen_index_cols = (2, 3, 4, 5)
eqtlgen_dataframe = pd.read_table(
    eqtlgen_input_file, header=0, index_col=eqtlgen_index_cols
)

edf_rows = eqtlgen_dataframe.index

be_overlaps = [x for x in bdf_rows if x in edf_rows]
be_overlap_dataframe = bios_dataframe.loc[be_overlaps, :]

print(be_overlap_dataframe.shape)
print(bios_dataframe.shape)
print(eqtlgen_dataframe.shape)
