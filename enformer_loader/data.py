#!/usr/bin/env python3

def get_chrom_sizes(chrom_sizes_file) -> dict:
    chrom_sizes = {}
    with open(chrom_sizes_file, "r") as f:
        for line in f:
            fields = line.split()
            chrom, size = fields[0], int(fields[1])
            if 'MT' not in chrom and 'KI' not in chrom and 'GL' not in chrom:
                chrom_sizes[chrom] = size
    return chrom_sizes
