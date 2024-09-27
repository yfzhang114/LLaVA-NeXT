from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric, Metric
from dataclasses import dataclass, field

from ..config import EvaluatorConfig, EvaluatorType, TrainConfig, DataConfig
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from ..torch_util import get_global_rank, get_world_size
from .downstream import ICLMetric, label_to_task_map
from .evaluator import Evaluator

__all__ = [
    "Evaluator",
    "ICLMetric",
    "label_to_task_map",
    "build_downstream_evaluator",
    "build_evaluator",
    "build_evaluators",
]


def build_downstream_evaluator(
    train_config,
    eval_cfg,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_kwargs = {}
    task_class = label_to_task_map[eval_cfg.label]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)  # type: ignore
    data_config = field(default_factory=DataConfig) # eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    else:
        ds_eval_sampler = DistributedSampler(
            ds_eval_dataset,
            drop_last=True, # drop_last
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            seed=train_config.seed,
        )
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=2,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=1,
        sampler=ds_eval_sampler,
        pin_memory=True,
        prefetch_factor=16,
        persistent_workers=True,
        timeout=0,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        label=eval_cfg.label,
        type=eval_cfg.type,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
        subset_num_batches=-1,
    )
    return evaluator


def build_evaluator(
    train_config, eval_config, tokenizer: Tokenizer, device: torch.device
) -> Evaluator:
    from ..data import build_eval_dataloader

    if eval_config.type == EvaluatorType.downstream:
        # Downstream evaluation.
        return build_downstream_evaluator(train_config, eval_config, tokenizer, device)
    elif eval_config.type == EvaluatorType.lm:
        # Language modeling evaluation.
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )

        def make_metric():
            return MeanMetric(nan_strategy="error").to(device)

        eval_metric: Union[Metric, Dict[str, Metric]]
        if eval_config.data.paths:
            eval_metric = make_metric()
        elif eval_config.data.datasets:
            eval_metric = {label: make_metric() for label in eval_config.data.datasets.keys()}
        else:
            raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")

        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")


def build_evaluators(cfg, device: torch.device) -> List[Evaluator]:
    evaluators = []
    tokenizer = cfg.tokenizer
    # tokenizer = Tokenizer.from_train_config(cfg)
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators
