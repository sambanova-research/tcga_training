"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import argparse
import sys
from typing import List

from torch import set_num_threads
from estimators.tcga_estimator import TCGAEstimator


def add_run_args(parser):
    parser.add_argument("--data-dir", type=str, help="root folder with images")
    parser.add_argument("--dataset-csv-path", type=str, help="dataset csv path.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adamw",
                        choices=["sgd", "adamw"],
                        help="pick between adamw and sgd")
    parser.add_argument('--in-height', type=int, default=512, help='Height of the input image')
    parser.add_argument('--in-width', type=int, default=512, help='Width of the input image')
    parser.add_argument('--channels', type=int, default=3, help='Width of the input image')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--drop-conv', type=float, default=0.03, help='dropout needed for conv layers of the model')
    parser.add_argument('--drop-fc', type=float, default=0.3, help='dropout needed for the fully connected layers')
    parser.add_argument('--model', type=str, default="rescale18",
                        choices=["rescale18", "rescale50"])
    parser.add_argument('--seed', type=int, default=256, help='Seed for the data')
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="initial lr (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for optimizer (default: 0.)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--print-freq", type=int, default=1, help="Frequency for logging metrics")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size of training and evaluation")
    parser.add_argument("--log-dir", type=str, default=None, help="Folder for logging")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Folder for saving checkpoints")
    parser.add_argument("--ckpt-file", type=str, default=None, help="Path to the checkpoint to be loaded")
    parser.add_argument("--inference", action="store_true", help="Flag for running inference")
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of samples loaded in advance by each dataloader worker.')
    parser.add_argument('--mode', type=str, default="train",
                        choices=["train", "validation", "predict"],
                        help='The mode to run the model')
    parser.add_argument('--local_rank', type=int, help='Only used by torch.distributed.launch command')
    parser.add_argument('--use-ddp', action='store_true',
                        help='Use Distributed Data Parallel for GPU training. Requires pytorch distributed launch.')


def main(argv: List[str]):
    # Set random seed for reproducibility.
    parser = argparse.ArgumentParser(description='Build model trainer')
    add_run_args(parser)
    args = parser.parse_args()

    # limit the number of threads spawned by torch pthreadpool_create to avoid
    # contention in the data loader workers.
    # the default is the number of cores in the system
    set_num_threads(1)

    estimator = TCGAEstimator(args, args.mode)

    if args.mode == "train":
        # launch model training
        estimator.train()
    elif args.mode == "validation":
        # run offline validation for all checkpoints and choose the best one
        if args.ckpt_dir is None:
            raise Exception("Need a ckpt dir to be specified when mode is validation")
        val_dataloader = estimator.get_test_dataloader(mode=args.mode)
        with open(f"{args.ckpt_dir}/checkpoints/checkpoint", "r") as f:
            checkpoints = f.readlines()
        all_val_auc = []
        all_ckpt = []
        for ckpt in checkpoints:
            if "latest" in ckpt:
                continue
            ckpt = ckpt.split(' ')[-1].rstrip("\n")
            all_ckpt.append(ckpt)
            ckpt_file = f"{args.ckpt_dir}/checkpoints/{ckpt}"
            val_metrics = estimator.validation(val_dataloader, ckpt_file)
            all_val_auc.append(val_metrics["auc"])
        all_val_auc = all_val_auc[::-1]
        all_ckpt = all_ckpt[::-1]
        max_auc = max(all_val_auc)
        best_ckpt = all_ckpt[all_val_auc.index(max_auc)]
        print(f"Max val auc: {max_auc}   Best checkpoint: {best_ckpt}")
    elif args.mode == "predict":
        # run inference and save artifacts for offline use
        estimator.evaluation("predict", args.ckpt_file)


if __name__ == '__main__':
    main(sys.argv[1:])
