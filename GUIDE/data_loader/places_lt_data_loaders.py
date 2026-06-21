
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from PIL import Image
import random
from dataset.others import aug_plus

class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = np.random.randint(0, self.bucket_num)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            np.random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num



class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                img_path_rel = line.split()[0]
                if img_path_rel.startswith('/'):
                    img_path_rel = img_path_rel[1:]

                self.img_path.append(os.path.join(root, img_path_rel))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class PlacesLTDataLoader(DataLoader):
    """
    Places-LT Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1,
                 training=True, balanced=False, retain_epoch_size=True,
                 train_txt="./data_txt/Places_LT/Places_LT_train.txt",
                 val_txt="./data_txt/Places_LT/Places_LT_val.txt",
                 test_txt="./data_txt/Places_LT/Places_LT_test.txt"):

        train_trsfm, test_trsfm = self.get_transformations()

        if training:
            dataset = LT_Dataset(data_dir, train_txt, train_trsfm)
            val_dataset = LT_Dataset(data_dir, val_txt, test_trsfm)
        else:  # test mode
            dataset = LT_Dataset(data_dir, test_txt, test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.n_samples = len(self.dataset)

        num_classes = 365
        self.cls_num_list = [0] * num_classes
        for label in dataset.targets:
            self.cls_num_list[label] += 1

        assert len(self.cls_num_list) == num_classes, "Error: Number of classes mismatch for Places-LT."

        if balanced:

            buckets = [[] for _ in range(num_classes)]
            for idx, label in enumerate(dataset.targets):
                buckets[label].append(idx)
            sampler = BalancedSampler(buckets, retain_epoch_size)
            shuffle = False  # Sampler and shuffle are mutually exclusive.
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler)

    def get_transformations(self):
        train_trsfm = aug_plus(dataset='ImageNet_LT', mode='train')
        test_trsfm = aug_plus(dataset='ImageNet_LT', mode='test')
        return train_trsfm, test_trsfm

    def split_validation(self):
        if self.val_dataset is None:
            return None
        else:
            return DataLoader(dataset=self.val_dataset, **self.init_kwargs)