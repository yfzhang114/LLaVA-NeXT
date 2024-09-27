from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size, is_distributed
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset, CombinedDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_memmap_dataset(
    paths, img_paths, img_mask_paths, effective_memmap_dtype, effective_memmap_img_dtype,
    datasets = None, 
    max_sequence_length: int = 2048, 
    max_num_images: int = 4,
    image_embedding_size: int = 1024,
    pad_token_id = 50256,
    generate_attention_mask = False,
    label_mask_paths = None,
    instance_filter = None,
    include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    img_paths: List[str]
    metadata: List[Dict[str, Any]] = []
    img_metadata: List[Dict[str, Any]] = []
    img_mask_metadata: List[Dict[str, Any]] = []
    if paths:
        for path in paths:
            metadata.append({"path": str(path)})
        for path in img_paths:
            img_metadata.append({"img_path": str(path)})
        for path in img_mask_paths:
            img_mask_metadata.append({"img_mask_path": str(path)})
    elif datasets:
        paths = []
        for label in sorted(datasets.keys()):
            label_paths = datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    text_dataset= MemMapDataset(
        *paths,
        chunk_size=max_sequence_length,
        memmap_dtype=effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=pad_token_id,
        generate_attention_mask=generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], label_mask_paths),
        instance_filter_config=instance_filter,
    )
    image_dataset = MemMapDataset(
        *img_paths,
        chunk_size=max_num_images * 576 * image_embedding_size,
        memmap_dtype=effective_memmap_img_dtype,
        metadata=img_metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=pad_token_id,
        generate_attention_mask=generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], label_mask_paths),
        instance_filter_config=instance_filter,
    )
    image_masks_dataset = MemMapDataset(
        *img_mask_paths,
        chunk_size=max_num_images,
        memmap_dtype=np.uint16,
        metadata=img_mask_metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=pad_token_id,
        generate_attention_mask=generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], label_mask_paths),
        instance_filter_config=instance_filter,
    )
    import pdb; pdb.set_trace()
    return CombinedDataset(text_dataset, image_dataset, image_masks_dataset)


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    # train_config: TrainConfig,
    paths,
    img_paths,
    img_mask_paths,
    *,
    save_folder = './',
    save_overwrite = True,
    epoch = None,
    global_train_batch_size = 512,
    seed = 6198,
    data_seed = None,
    drop_last = True,
    num_workers = 4,
    pin_memory = True,
    prefetch_factor = 4,
    persistent_workers = True,
    timeout = 0,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    device_train_batch_size = global_train_batch_size // 1
    collator = DataCollator(
        pad_direction='right', pad_token_id=50256
    )
    dataset = build_memmap_dataset(
        paths, img_paths, img_mask_paths, np.int32, np.float64, include_instance_metadata=include_instance_metadata
    )
    work_dir = Path(save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = data_seed if data_seed is not None else seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            global_train_batch_size,
            seed=seed + (epoch or 0),
            shuffle=True,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            fs_local_rank=fs_local_rank,
            work_dir=work_dir,
        ),
        batch_size=device_train_batch_size,
        drop_last=drop_last,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=False if num_workers == 0 else persistent_workers,
        timeout=timeout,
    )
