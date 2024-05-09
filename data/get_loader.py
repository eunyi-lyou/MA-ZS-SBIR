import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parent.resolve())
from .tuberlin.dataset import get_tuberlin_dataset
from .quickdraw.dataset import get_quickdraw_dataset
from .sketchy.dataset import get_sketchy_dataset

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor

data_dir_paths = [
    "tuberlin_dir_path",
    "quickdraw_dir_path",
    "sketchy_dir_path",
]

class CONCAT_DATASET:
    
    def __init__(self, args, ds_idx = None):
        super().__init__()
        
        self.ds_idx = ds_idx
        self.datasets = {}

        self.processor = AutoProcessor.from_pretrained(args.model.backbone)
        self.processor.tokenizer.model_max_length = args.model.max_length
        

    def _stack_datasets(self, split, cls_lists):
        fn_list = [get_tuberlin_dataset, get_quickdraw_dataset, get_sketchy_dataset]
        
        tmp_datasets = []
        for i in self.ds_idx:
            tmp_datasets.append(fn_list[i](split=split, cls_list=cls_lists[i], data_dir=data_dir_paths[i]))
        
        self.datasets = tmp_datasets
        
    def get_concat_dataset(self, split, cls_lists, partial_ratio=1.0):
        """
        split: train or valid
        partial_ratio: use part of dataset (instead of full)
        """
        self._stack_datasets(split, cls_lists)
        self.datasets = ConcatDataset(self.datasets)
        
        if partial_ratio < 1.0:
            partial_size = int(partial_ratio * len(self.datasets))
            self.datasets, _ = torch.utils.data.random_split(self.datasets, [partial_size, len(self.datasets) - partial_size])
        
        return self.datasets

    def _collate_fn(self, x):
        texts, images, mod_idxs, labels = [], [], [], []
        for datum in x:
            texts.append(datum["text"])
            images.append(datum["image"])
            mod_idxs.append(datum["mod_idx"][0])
            labels.append(datum["label"])
        
        out = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)

        out["mod_idx"] = torch.tensor(mod_idxs)
        out["label"] = torch.tensor(labels)
        return out
    

def get_loader(cfg, ds_idx, cls_lists, is_train, split='train'):
    
    partial_ratio=cfg.dataset.train_partial_ratio if is_train else cfg.dataset.valid_partial_ratio
    
    dataset_cls = CONCAT_DATASET(cfg, ds_idx = ds_idx)
    datasets = dataset_cls.get_concat_dataset(split=split, cls_lists=cls_lists, partial_ratio = partial_ratio)
    batch_size = cfg.train.batch_size
    
    loader = DataLoader(datasets, batch_size=batch_size, collate_fn=dataset_cls._collate_fn, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    return loader