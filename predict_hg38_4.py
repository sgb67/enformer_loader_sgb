#!/usr/bin/env python3

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import pysam
import pyfaidx
import tensorflow as tf

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dna

from enformer_helpers import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# the goal is to predict the tracks that Enformer was trained to predict, for the full sequence of a certain chromosome


# parameters ====================
# TODO can make this customisable
fasta_file = '/lustre/scratch126/gengen/projects/graft/recount_style_reference/hg38_galGal6_full/fasta/GRCh38.GRCg6a.full.renamed.merged.fa'
transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = '../enformer_dl_backup/c444fdff3e183daf686869692c26e00391f6773c/'
targets_txt = './targets_human.txt' # change to targets_mouse.txt if mouse
chrom_sizes_path='GRCh38.GRCg6a.full.renamed.merged.chrom.sizes.sorted'

chrom = 'hg38_4'

chrom_sizes = pd.read_csv(chrom_sizes_path, sep='\t', header=None, index_col=0)
chrom_sizes.columns = ['size']
chrom_length = chrom_sizes.loc[chrom].item()

pred_dir = f'./predictions/human/{chrom}'
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

prefix = '20241205'


# globals =======================
seq_len = 393216
# Padding on either side (only middle 196608 used), corresponding to 768 bins on either side.
OUTER_PRED_WIDTH = seq_len - (768 * 128) - (768 * 128)
# Only center 114688 basepairs are used, 320 bins cropped on either side.
PRED_WIDTH = OUTER_PRED_WIDTH - (320 * 128) - (320 * 128)
OUT_N_BINS = (PRED_WIDTH // 128)


# main fns ======================
def main():
    if not tf.config.list_physical_devices('GPU'):
        print('Running on CPU')
    else:
        print('GPU available')
    
    pyfaidx.Faidx(fasta_file)
    fasta_open = pysam.Fastafile(fasta_file)
    model = Enformer(model_path)
    targets_df = pd.read_csv(targets_txt, sep='\t')

    intervals = [(chrom, s, s+seq_len) for s in range(0, chrom_length - seq_len, PRED_WIDTH)]
    n_intervals = len(intervals)
    print(n_intervals)

    # Add targets to save here:
    # TODO read from JSON or YAML file or something.
    targets_predicted_list = {
        'HEK293_DNase': 'DNASE:HEK293T',
        'HEK293_CAGE': 'CAGE:embryonic kidney cell line: HEK293/SLAM untreated',
        'HEK293_H3K4me3': 'CHIP:H3K4me3:HEK293',
        'HEK293_H3K4me1': 'CHIP:H3K4me1:HEK293',
        'HEK293_H3K9me3': 'CHIP:H3K9me3:HEK293',
        'HEK293_H3K36me3': 'CHIP:H3K36me3:HEK293',
        'HEK293_eGFP-CTCF': 'CHIP:eGFP-CTCF:HEK293 genetically modified using site-specific recombination originated from HEK293',
        'K562_CAGE': 'CAGE:chronic myelogenous leukemia cell line:K562',
        'K562_DNase': 'DNASE:K562',
        'K562_CTCF': 'CHIP:CTCF:K562',
        'K562_H3K27me3': 'CHIP:H3K27me3:K562',
        'H1-hESC_DNase': 'DNASE:H1-hESC',
        'H1-hESC_H3K27ac': 'CHIP:H3K27ac:H1-hESC',
        'H1-hESC_H3K27me3': 'CHIP:H3K27me3:H1-hESC',
        'GM12878_DNase': 'DNASE:GM12878',
        'GM12878_CAGE': 'CAGE:B lymphoblastoid cell line: GM12878 ENCODE, biol_',
        'GM12878_H3K27ac': 'CHIP:H3K27ac:GM12878',
        'GM12878_H3K27me3': 'CHIP:H3K27me3:GM12878',
        'GM12878_CTCF': 'CHIP:CTCF:GM12878',
        'HELA-S3_H3K27ac': 'CHIP:H3K27ac:HeLa-S3',
        'HELA-S3_CTCF': 'CHIP:CTCF:HeLa-S3',
        'A549-ethanol_H3K27ac': 'CHIP:H3K27ac:A549 treated with 0.02% ethanol for 1 hour',
        'Kidney-epithelial-cell_H3K27me3': 'CHIP:H3K27me3:kidney epithelial cell',
        'Kidney-male-adult-50y_H3K27ac': 'CHIP:H3K27ac:kidney male adult (50 years)',
        'Kidney-male-adult-50y_H3K4me3': 'CHIP:H3K4me3:kidney male adult (50 years)'
    }

    targets_to_pred = {}
    for key, val in targets_predicted_list.items():
        targets_to_pred[key] = targets_df.loc[targets_df['description'] == val]['index'].tolist()

    # Check if all targets contain at least one track
    assert sum([len(targets_to_pred[key]) == 0 for key in targets_to_pred.keys()]) == 0, 'Some targets are not found'

    all_target_preds = dict(zip(
        targets_to_pred.keys(),
        [np.zeros((OUT_N_BINS * n_intervals, 1)) for i in range(len(targets_to_pred.keys()))]
    ))

    print(f'Predicting: {targets_to_pred}')

    curr_i = 0
    print(f'Batch 1/{len(intervals)}')
    for i, (c, s, e) in enumerate(intervals):
        if (i + 1) % 10 == 0:
            print(f'Batch {i+1}/{len(intervals)}')
        # One-hot encode the sequence of the current interval
        sequence_one_hot_wt = process_sequence(fasta_open, c, s, e)
        # Keep the mean of the model folds:
        predictions = predict_tracks(model, sequence_one_hot_wt)['human'][0]
        bs = predictions.shape[0]
        # Keep the mean signal in the targets of interest, store in all_predictions
        for target in targets_to_pred.keys():
            all_target_preds[target][curr_i:curr_i+bs, 0] = predictions[:, targets_to_pred[target]].mean(1)
        curr_i += bs

    all_pred_intervals = [(s, s+128) for s in range(intervals[0][1] + ((768 + 320) * 128), intervals[-1][2] - ((768 + 320) * 128), 128)]

    last_preds = dict(zip(
        targets_to_pred.keys(),
        [np.zeros((OUT_N_BINS, 1)) for i in range(len(targets_to_pred.keys()))]
    ))

    last_interval_start = ((chrom_length - seq_len) - ((chrom_length - seq_len) % 128))
    last_interval_end = last_interval_start + seq_len
    sequence_one_hot_wt = process_sequence(fasta_open, chrom, last_interval_start, last_interval_end)
    predictions = predict_tracks(model, sequence_one_hot_wt)['human'][0]
    bs = predictions.shape[0]

    # Make last predictions
    for target in targets_to_pred.keys():
        last_preds[target][0:bs, 0] = predictions[:, targets_to_pred[target]].mean(1)

    last_interval_pred_interval = [last_interval_start + ((768 + 320) * 128), last_interval_end - ((768 + 320) * 128)]
    last_interval_pred_intervals = [(s, s + 128) for s in range(last_interval_pred_interval[0], last_interval_pred_interval[1], 128)]
    indices_to_append = [i for i, x in enumerate(last_interval_pred_intervals) if x not in all_pred_intervals[-OUT_N_BINS:]]
    intervals_to_append = [(x,y) for (x,y) in np.array(last_interval_pred_intervals)[indices_to_append]]

    concat_preds = {}
    for target in targets_to_pred.keys():
        concat_preds[target] = np.concatenate((all_target_preds[target], last_preds[target][indices_to_append]))

    for target in concat_preds.keys():
        with open(f'{pred_dir}/{prefix}_enformer_fullpreds_{target}.bedGraph', 'w') as f:
            f.write(f'track type=bedGraph name="enformer predictions ({target})" description="enformer predictions ({target})"\n')
            for i, ((s, e), score) in enumerate(zip((all_pred_intervals + intervals_to_append), concat_preds[target][:,0])):
                f.write(f'{chrom}\t{s}\t{e}\t{score}\n')
    

    return 0


if __name__ == '__main__':
    main()

