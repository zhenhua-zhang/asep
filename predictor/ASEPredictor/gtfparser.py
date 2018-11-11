__version__ = '0.1'
__author__ = 'zhzhang(zhzhang2015@sina.com)'

import os
import json
import logging
from collections import defaultdict


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
FORMAT = '%(asctime)-15s\n  File:    %(name)s\n  Message: %(message)s\n'
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


class GTFParser:
    """ A simple API to parse GTF file
    """
    def __init__(self, fn):
        self.fileName = fn
        self.fnHandle = self._open(fn)
        # self.isGTF = self._isGTF(fn)
        # self.isGFF = self._isGFF(fn)
        self.GTFDict = self.GTF2Dict()


    def _isExistAndReadable(self, fn) -> int:
        pathExist = os.access(fn, os.F_OK)
        readable = os.access(fn, os.R_OK)
        
        if pathExist and readable:
            logger.info('File(%s) is existing and readable' %fn)
            return 1
        elif pathExist:
            logger.error('File(%s) is existing but not readable' %fn)
            return 0
        else:
            logger.error('File(%s) is neither existing nor readable' %fn)
            return 0

    def _isExistAndWritable(self, fn) -> int:
        pathExist = os.access(fn, os.F_OK)
        writable = os.access(fn, os.R_OK)

        if pathExist and writable:
            logger.info('File(%s) is existing and writable' %fn)
            return 1
        elif pathExist:
            logger.error('File(%s) is existing but not writable' %fn)
            return 0
        else:
            logger.error('File(%s) is neither existing nor writable' %fn)
            return 0

    def _open(self, fn):
        if self._isExistAndReadable(fn):
            logger.info("Successed to open file(%s)" %fn)
            return open(fn)
        logger.error("Failed to open file(%s)" %fn)
        return None


    def _isGTF(self, fn) -> int:
        logger.info('File(%s) is in GTF format' %fn)
        return 0


    def _isGFF(self, fn) -> int:
        logger.info('File(%s) is in GFF format' %fn)
        return 0

    def GTF2Dict(self) -> dict:
        fileHandler = self.fnHandle  # File handler of input file 
        GTFDict = defaultdict(bool)  # Dict including all GTF records
        line = next(fileHandler, "EOF")  # Each line of GTF records
        _lineCounter, _gene_id, _transcript_id = 0, '', ''
        while 1:
            _lineDict = {}
            _lineCounter += 1
            if line == 'EOF':
                    break

            if line.startswith('#'):
                line = next(fileHandler, "EOF")
                continue
            
            (_lineDict['chr'], _lineDict['source'], _lineDict['feature'],
             _lineDict['start'], _lineDict['stop'], _lineDict['score'],
             _lineDict['strand'], _lineDict['frame'], attribute 
            ) = line.split('\t')

            _attrList = attribute.strip().split(' ')
            if len(_attrList) % 2 != 0:
                logger.error('Bad "attribute" at line %s...'%_lineCounter)

            _attrDict = {
                x : y.replace('"', '').replace(';', '') 
                for x, y in zip(_attrList[0::2], _attrList[1::2])
            }

            gene_id = _attrDict['gene_id']
            if gene_id == _gene_id:
                transcript_id = _attrDict['transcript_id']
                if _transcript_id == transcript_id:
                    if 'protein_id' in attribute:
                        xxx_id = _attrDict['protein_id']
                        xxx_info = 'protein_info'

                    if 'ccds_id' in attribute:
                        xxx_id = _attrDict['ccds_id']
                        xxx_info = 'ccds_info'

                    if 'exon_id' in attribute:
                        xxx_id = _attrDict['exon_id']
                        xxx_info = 'exon_info'

                    GTFDict[gene_id][transcript_id][xxx_id] = {
                        xxx_info: _lineDict, 'attribute':_attrDict
                    }
                else:
                    GTFDict[gene_id][transcript_id] = {
                        'transcript_info': _lineDict, 'attribute': _attrDict
                    }

                _transcript_id = transcript_id
            else:
                GTFDict[gene_id] = {
                    'gene_info': _lineDict,
                    'attribute': _attrDict
                }
            _gene_id = gene_id

            line = next(fileHandler, "EOF")
        return GTFDict


    def GTF2Json(self):
        return json.dumps(self.GTFDict)
