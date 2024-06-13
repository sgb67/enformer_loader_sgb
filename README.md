# enformer_loader

This repo is made to preprocess `.bigWig` data to be used to finetune Enformer, and to provide Pytorch `DataLoader` structures to do so.

Currently only supports prediction of a single track.

## Acknowledgments
This repo builds primarily on the work done in the repo [`enformer-finetune`](https://github.com/wconnell/enformer-finetune/tree/c2145a628efcb91b932cc063a658e4a994bc4baa) by `wconnell` (License: Apache 2.0), and predicts using the [`enformer-pytorch`](https://github.com/lucidrains/enformer-pytorch) implementation of Enformer by `lucidrains` (License: MIT).

## Installation
1. Pull the repository and `cd` into the repo directory
2. Install using `pip install -e .`
3. Profit!

## Usage

### Creating a bed file with training data
Use the script `enformer_loader/scripts/generate_dataset.py` to generate a `.bed` file with data. 
If you are in the root directory of the repo, try running the following command to try it out:
```python
python enformer_loader/scripts/generate_dataset.py \
    tests/data/chrom_sizes.txt tests/data/test.bw 100 \
        tests/data/test_dataset.bed --n_bins 2 --bin_size 20 \
            --padding 20 --seed 1337 \
                --exclude_list_path tests/data/exclude_chroms.txt
```
Run `python enformer_loader/scripts/generate_dataset.py --help` to see what the arguments are.
Example output (first few lines):
```tsv
chr1	331	371	[20.0, 20.0]
chr1	266	306	[20.0, 20.0]
chr1	646	686	[20.0, 20.0]
chr1	739	779	[20.0, 20.0]
chr1	14	54	[20.0, 20.0]
chr1	596	636	[20.0, 20.0]
...
```

### Creating bed files with training and validation data
In this example scenario, I want to generate a training set that does not
contain chromosome 2, and a validation set with only chromosome 2.
Note this needs inputs of newline-separated text files that specify the 
chromosomes to be used in the validation set - and one in which you can specify
chromosomes to omit altogether. Also specify the number of sequences to be
included in the train and validation sets respectively, and the output files.

To see all arguments, use the following:
```bash
python enformer_loader/scripts/generate_train_val_dataset.py --help
```

To test the script, use the following:
```bash
python enformer_loader/scripts/generate_train_val_dataset.py \
    tests/data/chrom_sizes.txt tests/data/test.bw 40 10 \
        tests/data/test_train_dataset.bed tests/data/test_val_dataset.bed \
            tests/data/val_chroms.txt tests/data/exclude_chroms.txt \
                --n_bins 4 --bin_size 20 --padding 20 --seed 1337
```

For example outputs for training and validation sets, see the `example_outputs/`
directory. **Note that the output will be the same when `--seed` is set to the**
**same number**.

### `.bw` input
To create new test input, convert a `.bedGraph` file with `bedGraphToBigWig`
from `UCSC` tools: see [this guide](https://genome.ucsc.edu/goldenPath/help/bigWig.html).
