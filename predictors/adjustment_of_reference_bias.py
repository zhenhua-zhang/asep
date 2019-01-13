
import gzip
import pysam

from operator import itemgetter
# from operator import attrgetter
# from operator import methodcaller

"""

# Pick only auto and sexual chrommosomes
awk '{if ($1 ~ /^[1-9XY]+){print $0}} Homo_sapiens.GRCh37.71.gtf \
    > Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf

# Sort the gtf on by chrommosome(column 1) and coordination(column 4 and 5),
# then compress it by bgzip
cat Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf \
    | sort -k1,1d -k4,4n -k5,5n \
    | bgzip > Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf.bgz

# Index the compressed file by tabix, also can be done by `pysam.tabix_index()`
tabix -p gff Homo_sapiens.GRCh37.71.Chr1-22.X.Y.gtf.bgz

"""


class BiasAdjustment:
    """Adjust reference bias

    When mapping reads to reference genome(e.g. GRCh37), there is a bias that
    reference alleles have more chance be mapped and consequently the expect ratio of
    alternative-counts / reference-counts is less than 1;
    """

    def __init__(self, fn):
        """Initialization of BiasAdjustment
        """
        self.ifh = self.read_file(fn)  # Input file handle

    def __str__(self):
        """__str__() method; return specific strings"""
        return "BiasAdjustment"

    def read_file(self, fn, mode='r'):
        """Load the target file"""

        if fn.endswith('.gz') or fn.endswith('.bgz'):
            return gzip.open(fn, mode=mode)
        else:
            return open(fn, mode=mode)

    @staticmethod
    def is_indexed(file_name, suffix, check_time_stamp=False):
        """Check if the input file has updated index file"""
        older = None
        if check_time_stamp:
            file_timestamp = os.path.getmtime(file_name)
            file_index_timestamp = os.path.getmtime(file_name + suffix)
            older = file_timestamp < file_index_timestamp

        indexed = os.path.exists(file_name)
        return (indexed, older)

    def sort_tsv(self, file_name, with_patch=False, file_type=None,
                 chrom_col=None, pos_col=None):
        """Sort tab-separated file
        """
        if file_type in ['vcf', 'bed']:
            chrom_col = 0
            pos_col = 1
        elif file_type in ['gff', 'gtf', 'gff3']:
            chrom_col = 0
            pos_col = 3
        elif None not in [file_type, chrom_col, pos_col]:
            pass
        else:
            raise ValueError(
                '''Please leave file_type and specify chrom_col and pos_col.'''
            )

        header, lines, name_column = [], [], ''
        file_handler = self.read_file(file_name, mode='r')

        line = next(file_handler, 'EOF')
        assert line is not 'EOF', 'The file cannot be empty.'

        # {lambda x: len(x) for x in list(file_handler) if x.startswith('#')}

        while line != 'EOF':
            if line.startswith('#'):
                header.append(line)
            line = next(file_handler, 'EOF')

        lines = [line.split('\t') for x in list(file_handler)]
        lines = sorted(lines, key=itemgetter(chrom_col, pos_col))
        return iter(header.extend(lines))

    def make_index(self, file_name, file_type='fa'):
        """Make index file for input file"""
        if file_type == 'fa':
            self.is_indexed(file_name, '.fai')
            pysam.faidx(file_name)
        elif file_type in ['bam', 'cram']:
            self.is_indexed(file_name, '.bai')
            pysam.index(file_name)
        elif file_type in ['gff', 'bed', 'vcf']:
            self.is_indexed(file_name, '.tbi')
            pysam.tabix_index(file_name, preset=file_type)

    def parse_fasta(self, fasta_fn):
        """Parse fasta file"""
        with pysam.Fastafile(fasta_fn) as fafh:
            print(fafh.fetch('22', 1, 10))

    def get_gene_region(self, genes=[]):
        """Get the region of target gene"""
        for gene in genes:
            yield dict(
                chrom=gene.g_chrom(),
                start=gene.get_start(),
                end=gene.get_end()
            )

    def overall_p(self):
        """Expectation of global probability of reads including alternative allele"""
        gene_list = []
        region = self.get_gene_retion(genes=gene_list)
        gene_region = "{chrom}:{start}-{end}".format(**region)

        alt_allele_reads = 0
        ref_allele_reads = 0
        if ref_allele_reads != 0:
            return float(alt_allele_reads) / ref_allele_reads
        return None

    def expected_adjusted_alt_counts(self, ):
        """The adjusted expectation of alternative allele counts"""


ipf = '../misc/chr22.fa'
# adj = BiasAdjustment(ipf)
