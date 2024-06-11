import argparse
import pyBigWig
import pandas as pd
import numpy as np

from enformer_loader.utils import get_chrom_sizes, sum_bin, random_region


def main(chrom_sizes_file, bw_path, n_seq_train, n_seq_val, output_file_train,
         output_file_val, val_chroms, n_bins=896, bin_size=128,
         exclude_chroms=None):
    """
    Generates a dataset from a bigwig file. The dataset will contain n_seq_train
    sequences for training and n_seq_val sequences for validation. Argument
    exclude_chroms is a list of chromosomes to exclude from the dataset
    altogether, e.g. ['chrM', 'chrY'], or those to be used in the test set only.
    The argument val_chroms is a list of chromosomes to be used in the
    validation set only. The output file will contain the following columns:
    - chrom: chromosome
    - start: start position
    - end: end position
    - values: list of values in the region
    """
    SEQ_LEN = n_bins * bin_size

    bw_file = pyBigWig.open(bw_path)
    chrom_sizes = get_chrom_sizes(chrom_sizes_file)
    chrom_sizes = {k: v for k, v in chrom_sizes.items()
                   if k not in exclude_chroms}
    chr_to_i = {k: i for i, k in enumerate(chrom_sizes.keys())}
    sample_probs = [v / sum(chrom_sizes.values())
                    for k, v in chrom_sizes.items()]

    # Set sample probabilities for train and validation sets
    sample_probs_train = sample_probs.copy()
    for chrom in val_chroms:
        sample_probs_train[chr_to_i[chrom]] = 0
    sample_probs_train = [p / sum(sample_probs_train)
                          for p in sample_probs_train]
    sample_probs_val = sample_probs.copy()
    train_chroms = [k for k, v in chrom_sizes.items() if k not in val_chroms]
    for chrom in train_chroms:
        sample_probs_val[chr_to_i[chrom]] = 0
    sample_probs_val = [p / sum(sample_probs_val) for p in sample_probs_val]

    train_dataset = []
    for i in range(n_seq_train):
        found = False
        while not found:
            # Sample a region
            chrom, start, end, values = random_region(
                chrom_sizes, bw_file, sample_probs_train, SEQ_LEN)
            if np.any(np.isnan(values)):
                continue
            binned_values = sum_bin(values, n_bins)

            # Do not include regions where all values are 0
            if not np.all(np.array(binned_values) == 0) and chrom not in val_chroms:
                found = True
                row_data = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'values': binned_values
                }
                train_dataset.append(row_data)
    train_dataset = pd.DataFrame(train_dataset)
    train_dataset[['chrom', 'start', 'end', 'values']].to_csv(
        output_file_train, sep="\t", header=False, index=False)
    print(f"Train dataset saved to {output_file_train}")

    val_dataset = []
    for i in range(n_seq_val):
        found = False
        while not found:
            # Sample a region
            chrom, start, end, values = random_region(
                chrom_sizes, bw_file, sample_probs_val, SEQ_LEN)
            if np.any(np.isnan(values)):
                print(f'{chrom}:{start}-{end}')
                continue
            binned_values = sum_bin(values, n_bins)

            # Do not include regions where all values are 0
            if not np.all(np.array(binned_values) == 0) and chrom in val_chroms:
                found = True
                row_data = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'values': binned_values
                }
                val_dataset.append(row_data)
    val_dataset = pd.DataFrame(val_dataset)
    val_dataset[['chrom', 'start', 'end', 'values']].to_csv(
        output_file_val, sep="\t", header=False, index=False)
    print(f"Validation dataset saved to {output_file_val}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process some files and parameters.")
    parser.add_argument('chrom_sizes_file', type=str,
                        help='Path to the chromosome sizes file')
    parser.add_argument('bw_path', type=str,
                        help='Path to the .bigWig / .bw file')
    parser.add_argument('n_seq_train', type=int,
                        help='Number of sequences in train dataset')
    parser.add_argument('n_seq_val', type=int,
                        help='Number of sequences in validation dataset')
    parser.add_argument('output_file_train', type=str,
                        help='Path to the training set output file')
    parser.add_argument('output_file_val', type=str,
                        help='Path to the validation set output file')
    parser.add_argument('val_chroms_path', type=str,
                        help='Path to the file containing the list of chromosomes to use in the validation set')
    parser.add_argument('exclude_list_path', type=str,
                        help='Path to the file containing the list of chromosomes to exclude',
                        default=None)
    parser.add_argument('--n_bins', type=int, default=896,
                        help='Number of bins in prediction window (default: 896)')
    parser.add_argument('--bin_size', type=int, default=128,
                        help='Size of each bin (default: 128)')
    args = parser.parse_args()

    with open(args.val_chroms_path, "r") as f:
        val_chroms_list = [line.strip() for line in f]

    exclude_list = []
    if args.exclude_list_path is not None:
        with open(args.exclude_list_path, "r") as f:
            for line in f:
                exclude_list.append(line.strip())

    # TODO add option to set seed
    # TODO add option to generate non-overlapping sets: train, val, test
    # TODO add option to ignore blacklisted or masked regions
    # pass exclude_list as a parameter too
    main(chrom_sizes_file=args.chrom_sizes_file,
         bw_path=args.bw_path,
         n_seq_train=args.n_seq_train,
         n_seq_val=args.n_seq_val,
         output_file_train=args.output_file_train,
         output_file_val=args.output_file_val,
         val_chroms=val_chroms_list,
         n_bins=args.n_bins,
         bin_size=args.bin_size,
         exclude_chroms=exclude_list)
