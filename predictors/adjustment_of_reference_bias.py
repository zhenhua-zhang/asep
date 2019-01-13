
import gzip
import pysam
import bcftools


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

    @staticmethod
    def is_indexed(self, file_name, suffix, check_time_stamp=False):
        """Check if the input file has updated index file"""
        older = None
        if check_time_stamp:
            file_timestamp = os.path.getmtime(file_name)
            file_index_timestamp = os.path.getmtime(file_name + suffix)
            older = file_timestamp < file_index_timestamp

        indexed = os.path.exists(file_name)
        return (indexed, older)

    def read_file(self, fn):
        """Load the target file"""
        mode = 'r'
        openF = open

        if fn.endswith('.gz') or fn.endswith('.bgz'):
            openF = gzip.open
            mode = 'rb'

        with openF(fn, mode=mode) as fnh:
            return fnh

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
adj = BiasAdjustment(ipf)
