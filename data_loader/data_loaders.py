import logging
import functools
from pathlib import Path
from typing import Dict, List, Union

import torch
from typeguard import typechecked
from torch.utils.data import DataLoader
from zsvision.zs_utils import memcache

from zsvision.zs_data_structures import HashableDict, HashableOrderedDict
from data_loader.MSVD_dataset import MSVD
from data_loader.LSMDC_dataset import LSMDC
from data_loader.DiDeMo_dataset import DiDeMo
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.ActivityNet_dataset import ActivityNet
from data_loader.YouCook2_dataset import YouCook2
from data_loader.QuerYD_dataset import QuerYD
from data_loader.QuerYDSegments_dataset import QuerYDSegments
from data_loader.VaTeX_dataset import VaTeX

@functools.lru_cache(maxsize=64, typed=False)
def dataset_loader(
        text_dropout: float,
        fuse_captions: bool,
        spatial_feats: bool,
        use_zeros_for_missing: bool,
        challenge_mode: bool,
        eval_only: bool,
        task: str,
        data_dir: str,
        text_agg: str,
        text_feat: str,
        split_name: str,
        dataset_name: str,
        cls_partition: str,
        root_feat_folder: str,
        challenge_test_root_feat_folder: str,
        text_dim: int,
        num_test_captions: int,
        restrict_train_captions: int,
        logger: logging.Logger,
        max_tokens: Dict[str, int],
        raw_input_dims: HashableOrderedDict,
        feat_aggregation: HashableDict,
        distil_params: Union[None, Dict],
        training_file: Union[None, str],
        caption_masks: Union[None, str],
        ce_shared_dim: Union[None, int],
        **args,
):
    print(f"refreshing cache for {dataset_name} data loader [{split_name}]")
    kwargs = dict(
        task=task,
        data_dir=Path(data_dir),
        text_dim=text_dim,
        logger=logger,
        eval_only=eval_only,
        text_agg=text_agg,
        text_feat=text_feat,
        max_tokens=max_tokens,
        split_name=split_name,
        cls_partition=cls_partition,
        spatial_feats=spatial_feats,
        text_dropout=text_dropout,
        fuse_captions=fuse_captions,
        raw_input_dims=raw_input_dims,
        challenge_mode=challenge_mode,
        root_feat_folder=root_feat_folder,
        feat_aggregation=feat_aggregation,
        num_test_captions=num_test_captions,
        use_zeros_for_missing=use_zeros_for_missing,
        restrict_train_captions=restrict_train_captions,
        challenge_test_root_feat_folder=challenge_test_root_feat_folder,
        distil_params=distil_params,
        training_file=training_file,
        caption_masks=caption_masks,
        ce_shared_dim=ce_shared_dim,
        **args,
    )
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    elif dataset_name == "DiDeMo":
        dataset = DiDeMo(**kwargs)
    elif dataset_name == "ActivityNet":
        dataset = ActivityNet(**kwargs)
    elif dataset_name == "YouCook2":
        dataset = YouCook2(**kwargs)
    elif dataset_name == "QuerYD":
        dataset = QuerYD(**kwargs)
    elif dataset_name == "QuerYDSegments":
        dataset = QuerYDSegments(**kwargs)
    elif dataset_name == "VaTeX":
        dataset = VaTeX(**kwargs)
    return dataset


class ExpertDataLoader:

    @typechecked
    def __init__(
            self,
            eval_only: bool,
            fuse_captions: bool,
            challenge_mode: bool,
            use_zeros_for_missing: bool,
            trn_cat: int,
            text_dim: int,
            batch_size: int,
            num_workers: int,
            num_test_captions: int,
            task: str,
            data_dir: str,
            text_agg: str,
            text_feat: str,
            split_name: str,
            dataset_name: str,
            root_feat_folder: str,
            text_dropout: float,
            max_tokens: Dict[str, int],
            raw_input_dims: Dict[str, int],
            feat_aggregation: Dict[str, Dict],
            logger: logging.Logger,
            spatial_feats: bool = False,
            restrict_train_captions: int = 0,
            drop_last: bool = False,
            refresh_lru_cache: bool = False,
            cls_partitions: List[str] = ["train", "val", "tiny", "challenge"],
            challenge_test_root_feat_folder: str = "challenge",
            distil_params: Union[None, Dict] = None,
            training_file: Union[None, str] = None,
            caption_masks: Union[None, str] = None,
            ce_shared_dim: Union[None, int] = None,
    ):

        # Ensure that the dictionaries are hashable to allow use of caching
        raw_input_dims = HashableOrderedDict(raw_input_dims)
        feat_aggregation = HashableDict(feat_aggregation)
        if distil_params is not None:
            distil_params = HashableDict(distil_params)
        max_tokens = HashableDict(max_tokens)

        if refresh_lru_cache:
            logger.info("Explicitly refreshing dataloader and cuda cache")
            dataset_loader.cache_clear()
            torch.cuda.empty_cache()
            memcache.cache_clear()

        if trn_cat:
            raise NotImplementedError(f"Support for trn cat will need to be re-added")

        common_kwargs = dict(
            task=task,
            logger=logger,
            data_dir=data_dir,
            text_dim=text_dim,
            text_agg=text_agg,
            eval_only=eval_only,
            text_feat=text_feat,
            max_tokens=max_tokens,
            dataset_name=dataset_name,
            text_dropout=text_dropout,
            fuse_captions=fuse_captions,
            spatial_feats=spatial_feats,
            split_name=split_name,
            challenge_mode=challenge_mode,
            root_feat_folder=root_feat_folder,
            use_zeros_for_missing=use_zeros_for_missing,
            challenge_test_root_feat_folder=challenge_test_root_feat_folder,
            num_test_captions=num_test_captions,
            raw_input_dims=raw_input_dims,
            feat_aggregation=feat_aggregation,
            restrict_train_captions=restrict_train_captions,
            distil_params=distil_params,
            training_file=training_file,
            caption_masks=caption_masks,
            ce_shared_dim=ce_shared_dim,
        )

        if "retrieval" in task:
            dataset = dataset_loader(cls_partition="train", **common_kwargs)
            x = dataset_loader.cache_info()  # pylint: disable=no-value-for-parameter
            logger.info(f"cache info {x}")
            self.dataloaders = {"dataset": dataset}
            self.dataloaders["retrieval"] = dataset.get_retrieval_data()
            if not eval_only:
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_data,
                    drop_last=drop_last,
                    shuffle=True,
                )
                self.dataloaders["train"] = train_loader
        else:
            self.dataloaders = {}
            for cls_partition in cls_partitions:
                cls_dataset = dataset_loader(cls_partition=cls_partition, **common_kwargs)
                x = dataset_loader.cache_info()  # pylint: disable=no-value-for-parameter
                logger.info(f"cache info [{cls_partition}] {x}")
                loader = DataLoader(
                    dataset=cls_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=cls_dataset.collate_data,
                    drop_last=False,
                    shuffle=False,
                )
                self.dataloaders[cls_partition] = loader

        logger.info(f"Loading data loaders with {num_workers} workers")
        self.num_test_captions = num_test_captions
        self.dataset_name = dataset_name

    def __getitem__(self, key):
        return self.dataloaders[key]
