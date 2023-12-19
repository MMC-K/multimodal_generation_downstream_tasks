#!/usr/bin/env python
# coding=utf-8

# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from curses import raw
import json
import logging
import math
import os
import random
from datetime import timedelta
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedDataParallelKwargs
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    ViTFeatureExtractor,
    SchedulerType,
    get_scheduler,
    default_data_collator,
)

import datasets
from datasets import load_dataset
import evaluate

from data_utils import DatasetForVLAlign
from modeling_veldt5 import VELDT5Model

from mecab import MeCab
from PIL import Image


logger = get_logger(__name__)

# epochs=1
# learning_rate=0.001
# scheduler_type=linear
# accelerate launch training_veldt5_accelerate.py \
# --vision_model 'google/vit-base-patch16-384' \
# --language_model 'KETI-AIR/ke-t5-base' \
# --gradient_accumulation_steps 32 \
# --per_device_train_batch_size 16 \
# --per_device_eval_batch_size 16 \
# --warmup_portion 0.02 \
# --logging_steps 20 \
# --checkpointing_steps 10000 \
# --num_train_epochs $epochs \
# --lr_scheduler_type $scheduler_type \
# --with_tracking \
# --output_dir veld_e${epochs}_${scheduler_type}


# accelerate launch training_veldt5_accelerate.py \
#     --max_train_steps_per_epoch 100 \
#     --max_validation_steps 20 \
#     --logging_steps 5 \
#     --with_tracking \
#     --output_dir test


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    # data
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name_lm",
        type=str,
        default="sent_dataset.py",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name_lm",
        type=str,
        default="base",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="../huggingface_datasets",
        help="The path to cache directory for huggingface datasets.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=1,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=256,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    
    # parser.add_argument("--train_path",
    #                     default="../../downloaded_data/train-filtered.json", type=str)
    parser.add_argument("--validation_path",
                        default="../../downloaded_data/validation-filtered.json", type=str)
    
    # parser.add_argument("--image_root_dir",
    #                     default="../../downloaded_data", type=str)

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="image_text_pair_datasets.py",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="base",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_data_dir",
        type=str,
        default="../../downloaded_data",
        help="The path to data directory for huggingface datasets.",
    )



    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    # training
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--contrastive_weight", default=1.0,
                        type=float, help="The weighting value for contrastive loss")
    parser.add_argument("--captioning_weight", default=2.0,
                        type=float, help="The weighting value for captioning loss")
    parser.add_argument("--lm_weight", default=1.0,
                        type=float, help="The weighting value for lm loss")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--logit_temperature", default=1.0,
                        type=float, help="temperature for logits")
    parser.add_argument("--label_smoothing", default=0.0,
                        type=float, help="label smoothing for cross entropy")
    # parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    # )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Total number of validation steps to perform.",
    )
    # parser.add_argument(
    #     "--max_train_steps_per_epoch",
    #     type=int,
    #     default=None,
    #     help="The number of training steps to perform on a epoch. (for debugging)",
    # )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    # parser.add_argument(
    #     "--warmup_portion", type=float, default=0, help="Portion of total training steps for the warmup in the lr scheduler."
    # )
    # parser.add_argument(
    #     "--checkpointing_steps",
    #     type=str,
    #     default=None,
    #     help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    # )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )


    # logging
    # parser.add_argument(
    #     "--logging_steps", type=int, default=0, help="Number of steps for logging (stdout)."
    # )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--from_veld_model",
        type=str,
        default=None,
        help=(
            "Path to model that you want to test"
        ),
    )

    parser.add_argument(
        "--save_caption_result",
        action="store_true",
        help="save caption results in <model_path>/figures/<img_num>.png and <model_path>/figures/captions.json",
    )

    args = parser.parse_args()
    print("[BG] args.validation_path:", args.validation_path)
    # assert(False)
    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        # accelerator_log_kwargs["logging_dir"] = args.output_dir

    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(days=10)),
        DistributedDataParallelKwargs(find_unused_parameters=True)
        ]

    # accelerator_log_kwargs["project_dir"] = accelerator_log_kwargs["logging_dir"] 
    # del accelerator_log_kwargs["logging_dir"] 
    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps, 
        kwargs_handlers=kwargs_handlers , **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
    

    model = None
    # Load model and tokenizer
    logger.info("***** Running from a pretrained VELD model *****")
    model = VELDT5Model.from_pretrained(args.from_veld_model)
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=not args.use_slow_tokenizer)


    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.


    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

      # load image text pair datasets
    with accelerator.main_process_first():
        image_text_datasets = load_dataset(
                args.dataset_name, 
                args.dataset_config_name,
                cache_dir=args.hf_cache_dir, 
                data_dir=args.hf_data_dir,
                # train_csv_path = args.train_path,
                validation_csv_path = args.validation_path
            )
    eval_dataset = image_text_datasets["validation"]
    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        image_list = [s["image"] for s in samples]
        image_feature = image_tokenizer(images=image_list, return_tensors="pt")
        text_feature = tokenizer([s["description"] for s in samples], return_tensors="pt", padding=True, truncation='longest_first')
        return {
            "pixel_values": image_feature["pixel_values"],
            "input_ids": text_feature["input_ids"],
            "attention_mask": text_feature["attention_mask"],
        }


    # Log a few random samples from the training set:
    print("[*] eval dataset json [0]", json.load(open(args.validation_path, mode="r", encoding='utf-8'))[0])
    print("[*] eval_dataset[0]", eval_dataset[0])
    # DataLoaders creation:
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)


    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # completed_steps = 0
    # Potentially load in the weights and states from a previous save
    # calc mrr
    def calc_mrr(encoder_logits, decoder_logits):
        with torch.no_grad():
            # outputs: language_repr, vision_repr- [batch_size, model_dim]
            encoder_logits = nn.functional.normalize(encoder_logits)
            decoder_logits = nn.functional.normalize(decoder_logits)

            batch_size = encoder_logits.size(0)
            scores = torch.mm(decoder_logits, encoder_logits.t())
            target = torch.arange(batch_size).to(decoder_logits.device)

            # scores: [batch_size, batch_size]
            ranked = scores.argsort(dim=1, descending=True)
            # [[0.1, 0.3, -0.2, 0.14 ]] -> [[1, 3, 0, 2]] (index of score - descending order)
            idx2ranked_t = ranked.argsort(dim=1)

            # [[1, 3, 0, 2]] -> [[2, 0, 3, 1]] (index to rank)
            rrs = []
            for t, idx2ranked in zip(target, idx2ranked_t):
                rrs.append(1 / (idx2ranked[t].item() + 1))
            
            # reciprocal rank
            return torch.tensor(np.mean(rrs)).to(decoder_logits.device)


    # evaluation proc
    model.eval()
    total_eval_loss = 0
    total_eval_mrr = 0
    unwrapped_model = accelerator.unwrap_model(model)
    MAX_NEW_TOKENS = 40
    MAX_ORDER = 1
    mecab = MeCab()
    # sacrebleu does not use custom tokenizer in compute() function
    #bleu = evaluate.load("sacrebleu", max_order=MAX_ORDER)
    bleu = evaluate.load("bleu", max_order=MAX_ORDER)
    rouge = evaluate.load("rouge")
    
    SAVE_CAPTION_DIR = os.path.join(args.from_veld_model, "figures")

    if args.save_caption_result and not os.path.exists(SAVE_CAPTION_DIR):
        os.makedirs(SAVE_CAPTION_DIR, exist_ok=True)
    
    capntion_result_list = []
    image_index = 0
    for step, batch in enumerate(eval_dataloader):
        if args.max_validation_steps is not None and step >= args.max_validation_steps:
            break
        
        with torch.no_grad():
            outputs = model(
                pixel_values=batch["pixel_values"], 
                labels=batch["input_ids"], 
                return_contrastive_loss=True,
                decoder_attention_mask=batch["attention_mask"], 
            )
            loss = args.captioning_weight*outputs.loss + args.contrastive_weight*outputs.c_loss
            mrr = calc_mrr(outputs.e_logits_g, outputs.d_logits)

            total_eval_loss += accelerator.reduce(loss).detach().float()
            total_eval_mrr += accelerator.reduce(mrr).detach().float()

            gen_seq_batch = unwrapped_model.generate(
                    pixel_values=batch["pixel_values"], 
                    num_beams=4,
                    num_return_sequences=1,
                    max_new_tokens=MAX_NEW_TOKENS
                )

            predictions = tokenizer.batch_decode(gen_seq_batch, skip_special_tokens=True)
            predictions = [p if len(p) > 0 else "<empty>" for p in predictions]
            references = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            if args.save_caption_result:
                for pv, p, r in zip(batch["pixel_values"], predictions, references) :
                    capntion_result_list.append({'image_index':image_index, 'reference':r, 'prediction':p})
                    im = Image.fromarray(np.clip((pv.cpu().permute(1, 2, 0).numpy()*255*1.2), 0, 255).astype(np.uint8))
                    im.save(os.path.join(SAVE_CAPTION_DIR, f"{image_index}.png"))
                    image_index += 1


            print("[*] References", references)
            print("[*] Prediction", predictions)
            
            bleu.add_batch(predictions=predictions, references=[[r] for r in references])
            rouge.add_batch(predictions=predictions, references=[[r] for r in references])
            
            if step > 25:
                break 

    if args.save_caption_result:
        f = open(os.path.join(SAVE_CAPTION_DIR, "captions.json"), mode="w")  #, encoding='utf-8'
        json.dump(capntion_result_list,f, indent=4, ensure_ascii = False)
        f.close()

    
    if accelerator.is_local_main_process:
        bleu_result = bleu.compute(tokenizer=lambda x: mecab.morphs(x), max_order=MAX_ORDER)
        rouge_result = rouge.compute(tokenizer=lambda x: mecab.morphs(x))
        bleu_result["bleu"] = bleu_result["bleu"]*100
        print(f"[*] bleu_result (100, MAX_ORDER {MAX_ORDER})", bleu_result)
        print("[*] rouge_result (1)", rouge_result)
    logger.info("Evaluation - loss: {}, mrr: {}".format(
            total_eval_loss.item() / accelerator.num_processes / len(eval_dataloader),
            total_eval_mrr.item() / accelerator.num_processes / len(eval_dataloader),
        ))
    


if __name__ == "__main__":
    main()

