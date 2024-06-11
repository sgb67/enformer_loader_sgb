import enformer_loader as efl


def test_get_chrom_sizes():
    chrom_sizes = efl.get_chrom_sizes("tests/data/chrom_sizes.txt")
    assert chrom_sizes == {'chr1': 123, 'chr2': 556}


def test_avg_bin():
    array = [1, 2, 3, 4, 5, 6]
    assert efl.avg_bin(array, 2) == [2.0, 5.0]


def test_sum_bin():
    array = [1, 2, 3, 4, 5, 6]
    assert efl.sum_bin(array, 2) == [6, 15]
