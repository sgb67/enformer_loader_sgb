import enformer_loader as efl


def test_genome_data_interval_dataset():
    all_chroms = ['chr1', 'chr2', 'chr3']
    bed_file_path = 'tests/data/test_train_dataset.bed'
    fasta_file_path = 'tests/data/sim_genome.fa'
    dataset = efl.GenomeDataIntervalDataset(
        all_chroms, bed_file_path, fasta_file_path)
    assert len(dataset) == 40
    assert dataset.check_tensor_dtype(dataset[0][1]) == 'torch.float32'
    assert dataset[0][0].shape == (80, 4)
    assert dataset[0][1].shape == (4, 1)
    chrom = dataset.int_to_chr[dataset[0][2][0].item()]
    assert chrom in all_chroms
