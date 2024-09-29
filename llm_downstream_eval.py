"""Run this script with 'torchrun'."""

import gzip
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Optional, TextIO
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, TextIO, Tuple, Union
import torch
import torch.distributed as dist
import transformers
from tqdm import tqdm
import wandb
from packaging import version

from llava.model import *
from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)

from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.torch_util import (
    barrier,
    get_default_device,
    move_to_device,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

import flash_attn
from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

log = logging.getLogger("")


@dataclass
class ModelArguments:
    base_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct")

@dataclass
class EvaluatorConfig:
    label: str
    type: str

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # List of Downstream 
    evaluators: List[EvaluatorConfig] = field(default_factory=lambda: [
        EvaluatorConfig(label="mmlu_stem_test", type="downstream"),
        EvaluatorConfig(label="mmlu_cloze_humanities", type="downstream"),
        EvaluatorConfig(label="mmlu_cloze_social_sciences", type="downstream"),
        EvaluatorConfig(label="mmlu_cloze_other", type="downstream"),
        EvaluatorConfig(label="sciq", type="downstream"),
        EvaluatorConfig(label="arc_easy", type="downstream"),
        EvaluatorConfig(label="piqa", type="downstream"),
        EvaluatorConfig(label="winogrande", type="downstream"),
        EvaluatorConfig(label="openbook_qa", type="downstream"),
        EvaluatorConfig(label="copa", type="downstream"),
        EvaluatorConfig(label="rte", type="downstream"),
        EvaluatorConfig(label="commitment_bank", type="downstream"),
        EvaluatorConfig(label="mrpc", type="downstream"),
        EvaluatorConfig(label="sst2", type="downstream"),
        EvaluatorConfig(label="hellaswag", type="downstream"),
    ])


def get_labels(batch: Dict[str, Any]) -> torch.Tensor:
    # Labels are just input IDs shifted to the left (first item is ignored).
    labels, label_mask, attention_mask, instance_mask = (
        batch["input_ids"].clone(),
        batch.get("label_mask"),
        batch.get("attention_mask"),
        batch.get("instance_mask"),
    )
    if label_mask is not None:
        labels.masked_fill_(~label_mask, -100)
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, -100)
    if instance_mask is not None:
        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
    return labels[..., 1:].contiguous()


def fused_loss_fn(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
        ):
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")
        if ce_loss_use_ignore_index_param:
            ignore_index_kwarg = {"ignore_index": ignore_index}
        else:
            ignore_index_kwarg = {"ignored_index": ignore_index}

        loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=0.0,
                    inplace_backward=False,
                    process_group=None,
                    **ignore_index_kwarg,
                )

        mask = labels != ignore_index

        if reduction == "mean":
                    loss = loss.sum() / mask.sum()
        elif reduction == "sum":
                    loss = loss.sum()
        else:
                    loss = loss

        if not compute_z_loss:
                return loss, None

        if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
                    z_loss = z_loss.sum()
        else:
                    z_loss = z_loss

        return loss, z_loss


def model_forward(
        model, batch: Dict[str, Any], loss_reduction: str = "mean", compute_z_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    # shape: (batch_size, seq_len, vocab_size)
    logits = model(
        input_ids=batch["input_ids"],
            # attention_mask=batch.get("attention_mask"),
    ).logits
    logits_for_loss = logits[..., :-1, :].contiguous()
    # shape: (batch_size * seq_len, vocab_size)
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
    # shape: (batch_size, seq_len)
    labels = get_labels(batch)
    # shape: (batch_size * seq_len,)
    labels = labels.view(-1)
    ce_loss, z_loss = fused_loss_fn(
        logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss
    )
    if loss_reduction == "none":
        # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
        if z_loss is not None:
            z_loss = z_loss.view(batch["input_ids"].shape[0], -1)
    return ce_loss, z_loss, logits


def eval_batch( batch: Dict[str, Any], model) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", enabled=True): # dtype=
        ce_loss, _, logits = model_forward(model, batch, loss_reduction="none")
    return ce_loss.mean(dim=-1), logits


def eval_step(batch: Dict[str, Any], evaluator, model, device) -> None:
    # Move tensors to the right device.
    batch = move_to_device(batch, device)

    # Run forward pass.
    with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
        ce_loss, logits = eval_batch(batch, model)

    # Update metrics.
    evaluator.update_metrics(
        batch, ce_loss, logits
    )  # batch includes all keys that the downstream evaluation needs


def eval(cfg, evaluators, model) -> Dict[str, Any]:
    model.eval()
    model.to(cfg.device)

    eval_metrics = {}
    for evaluator in evaluators:
        log.info(f"Running evaluation for '{evaluator.label}'...")

        # Reset metrics.
        evaluator.reset_metrics()

        # Initialize data loader iterator.
        eval_batches = iter(evaluator.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = (
            evaluator.subset_num_batches
            if evaluator.subset_num_batches is not None
            else -1 # eval_subset_num_batches
        )
        
        if num_eval_batches > 0:
            num_eval_batches = min(num_eval_batches, len(evaluator.eval_loader))
            eval_batches = islice(eval_batches, num_eval_batches)


        # Run model over batches.
        for i_step, eval_batch in tqdm(enumerate(eval_batches)):

            eval_step(eval_batch, evaluator, model, cfg.device)

            # Log to console.
            console_log_interval = 1
            if i_step + 1 == num_eval_batches or (i_step + 1) % console_log_interval == 0:
                log.info(f"[eval_step={i_step + 1}/{num_eval_batches}]")

        # Get final metrics.
        metrics = evaluator.compute_metrics()
        new_metrics = {}
        for key, value in metrics.items():
            new_key = evaluator.label
            count = 1
            new_metrics[new_key] = {'Acc': value, 'Cnt': i_step}

        eval_metrics.update(new_metrics)
        print(new_metrics)

        del eval_batches

    return eval_metrics


def main(attn_implementation='flash_attention_2') -> None:
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name_or_path)
    training_args.tokenizer = tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_args.base_model_name_or_path) 
    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    # Construct evaluators.
    evaluators = build_evaluators(training_args, device)
    barrier()

    eval_metrics = eval(training_args, evaluators, model)
    print(eval_metrics)
    
    total_acc = 0
    total_cnt = 0
    weighted_acc_sum = 0

    for metric in eval_metrics.values():
        acc = metric['Acc']
        cnt = metric['Cnt']
        total_acc += acc
        total_cnt += 1  # 计算总指标数
        weighted_acc_sum += acc * cnt  # 加权和

    # 计算平均 ACC
    average_acc = total_acc / total_cnt if total_cnt > 0 else 0

    # 计算加权平均 ACC
    weighted_average_acc = weighted_acc_sum / total_cnt if total_cnt > 0 else 0

    # 输出结果
    print(f"Average Accuracy: {average_acc:.2f}")
    print(f"Weighted Average Accuracy: {weighted_average_acc:.2f}")

if __name__ == "__main__":
    # Initialize process group.
    import random
    seed = 3407
    torch.manual_seed(seed)
    random.seed(seed)
    main()
