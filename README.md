# Enformer_loader

This repo is made to preprocess `.bigWig` data to be used to finetune Enformer, and to provide Pytorch `DataLoader` structures to do so. Currently only supports prediction of a single track.

## Acknowledgments
This repo builds upon the repo `enformer-finetune` at https://github.com/wconnell/enformer-finetune/tree/c2145a628efcb91b932cc063a658e4a994bc4baa

## Installation
1. Pull the repository and `cd` into the repo directory
2. Install using `pip install -e .`
3. Profit!

## Usage

### Creating a bed file with training data
Use the script `enformer_loader/scripts/generate_dataset.py` to generate a `.bed` file with data. 
If you are in the root directory of the repo, try running the following command to try it out:
```
python enformer_loader/scripts/generate_dataset.py tests/data/chrom_sizes.txt tests/data/test.bw 100 example_outputs/test.bed --n_bins 2 --bin_size 20
```
Run `python enformer_loader/scripts/generate_dataset.py --help` to see what the arguments are.
Example output (first few lines):
```
chr1	331	371	[20.0, 20.0]
chr1	266	306	[20.0, 20.0]
chr1	646	686	[20.0, 20.0]
chr1	739	779	[20.0, 20.0]
chr1	14	54	[20.0, 20.0]
chr1	596	636	[20.0, 20.0]
...
```


