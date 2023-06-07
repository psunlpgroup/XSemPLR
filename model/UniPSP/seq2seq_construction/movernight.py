from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = MOvernightDataset(self.args, raw_datasets['train'])
        dev_dataset = MOvernightDataset(self.args, raw_datasets['validation'])
        test_dataset = MOvernightDataset(self.args, raw_datasets['test'])

        return train_dataset, dev_dataset, test_dataset


class MOvernightDataset(Dataset):

    def __init__(self, args, raw_datasets):
        self.args = args

        self.extended_data = []
        for raw_data in raw_datasets:
            question = raw_data["question"][self.args.dataset.language]
            mr = raw_data["mr"][self.args.dataset.mr]
            raw_data.update({"struct_in": "",
                             "text_in": question,
                             "seq_out": mr})
            self.extended_data.append(raw_data)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
