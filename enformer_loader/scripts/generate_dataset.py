import argparse
import pyBigWig
import pandas as pd
import numpy as np

from enformer_loader.utils import get_chrom_sizes, sum_bin, random_region, avg_bin


def main(chrom_sizes_file, bw_path, n_seq, output_file, exclude_list_path=None,
         n_bins=896, bin_size=128, padding=0, seed=1337, use_sum=False):
    np.random.seed(seed)

    SEQ_LEN = n_bins * bin_size

    exclude_list = []
    if exclude_list_path is not None:
        with open(exclude_list_path, "r") as f:
            exclude_list = [line.strip() for line in f]

    bw_file = pyBigWig.open(bw_path)
    chrom_sizes = get_chrom_sizes(chrom_sizes_file)
    chrom_sizes = {k: v for k, v in chrom_sizes.items()
                   if k not in exclude_list}
    sample_probs = [v / sum(chrom_sizes.values())
                    for k, v in chrom_sizes.items()]

    dataset = []
    for i in range(n_seq):
        found = False
        while not found:
            # Sample a region
            chrom, start, end, values = random_region(
                chrom_sizes, bw_file, sample_probs, SEQ_LEN, padding=padding)
            if np.any(np.isnan(values)):
                continue
            if use_sum:
                binned_values = sum_bin(values, n_bins)
            else:
                binned_values = avg_bin(values, n_bins)

            # Do not include regions where all values are 0
            if not np.all(np.array(binned_values) == 0):
                found = True
                row_data = {
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'values': binned_values
                }
                dataset.append(row_data)
    dataset = pd.DataFrame(dataset)
    dataset[['chrom', 'start', 'end', 'values']].to_csv(
        output_file, sep="\t", header=False, index=False)
    print(f"Dataset saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process some files and parameters.")
    parser.add_argument('chrom_sizes_file', type=str,
                        help='Path to the chromosome sizes file')
    parser.add_argument('bw_path', type=str,
                        help='Path to the .bigWig / .bw file')
    parser.add_argument('n_seq', type=int,
                        help='Number of sequences in dataset')
    parser.add_argument('output_file', type=str,
                        help='Path to the output file')
    parser.add_argument('--exclude_list_path', type=str,
                        help='Path to the file containing the list of chromosomes to exclude',
                        default=None)
    parser.add_argument('--n_bins', type=int, default=896,
                        help='Number of bins in prediction window (default: 896)')
    parser.add_argument('--bin_size', type=int, default=128,
                        help='Size of each bin (default: 128)')
    parser.add_argument('--padding', type=int, default=0,
                        help='Amount of padding to add to each side of the region later (default: 0)')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed (default: 1337)')
    parser.add_argument('--use_sum', action='store_true',
                        help='Sum within each bin instead of average')

    # TODO add option to ignore blacklisted or masked regions
    args = parser.parse_args()
    main(*vars(args).values())
