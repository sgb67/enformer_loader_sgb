import ast
import torch
from enformer_pytorch import GenomeIntervalDataset


class GenomeDataIntervalDataset(GenomeIntervalDataset):
    def __init__(self, all_chroms, *args, **kwargs):
        """
        Argument all_chroms is a list of chromosomes in the dataset.
        E.g. ['chr1', 'chr2', ...]
        """
        super(GenomeDataIntervalDataset, self).__init__(*args, **kwargs)
        self.chr_to_int = dict(zip(all_chroms, range(len(all_chroms))))
        self.int_to_chr = {v: k for k, v in self.chr_to_int.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, target = (
            interval[0], interval[1], interval[2], interval[3])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target).unsqueeze(-1)
        sequence = self.fasta(chr_name, start, end,
                              return_augs=self.return_augs)
        loc = torch.tensor([self.chr_to_int[chr_name], start, end])
        return sequence, target, loc

    @staticmethod
    def check_tensor_dtype(tensor):
        if tensor.dtype == torch.float16:
            return "fp16"
        elif tensor.dtype == torch.bfloat16:
            return "bf16"
        else:
            return str(tensor.dtype)
