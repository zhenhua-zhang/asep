#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open('phenotyped_by_trityper_noLL.txt', 'r') as pFile:
    pFileDict = {}
    line = next(pFile)
    while 1:
        lineList = line.strip().split("\t")
        pFileDict[lineList[1]] = lineList[0]
        line = next(pFile, 'EOF')
        if line == 'EOF': 
            break

    pFileKeypool = pFileDict.keys()
    with open('freeze2_complete_GTE_Groningen_07092016.txt', 'r') as fFile:
        fLine = next(fFile)
        while 1:
            lineDNA, lineRNA = fLine.strip().split('\t')
            if lineDNA in pFileKeypool:
                fLine += pFileDict[lineDNA]
                print(fLine.replace("\n", "\t"))
            fLine = next(fFile, 'EOF')
            if fLine == 'EOF':
                break

