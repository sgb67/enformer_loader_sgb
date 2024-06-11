import enformer_loader as efl
import pyBigWig


def test_get_chrom_sizes():
    chrom_sizes = efl.get_chrom_sizes("tests/data/chrom_sizes.txt")
    assert chrom_sizes == {'chr1': 1000, 'chr2': 556}


def test_avg_bin():
    array = [1, 2, 3, 4, 5, 6]
    assert efl.avg_bin(array, 2) == [2.0, 5.0]


def test_sum_bin():
    array = [1, 2, 3, 4, 5, 6]
    assert efl.sum_bin(array, 2) == [6, 15]


def test_random_region():
    """
    This test is not deterministic, but it should pass most of the time if not
    all, since the chr1 is sampled with probability 1, and the signal across
    the region is 1.0, so the sum of the values should be 100.0.
    """
    chrom_sizes = efl.get_chrom_sizes("tests/data/chrom_sizes.txt")
    bw_file_path = "tests/data/test.bw"
    bw_file = pyBigWig.open(bw_file_path)
    chrom, start, end, values = efl.random_region(
        chrom_sizes, bw_file, p=[1, 0], SEQ_LEN=100)
    print(f'{chrom}:{start}-{end}')
    assert chrom in chrom_sizes.keys()
    assert start >= 0
    assert end <= chrom_sizes[chrom]
    assert len(values) == 100
    assert sum(values) == 100.0
