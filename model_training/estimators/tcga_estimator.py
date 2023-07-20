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
import copy
import collections
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T
from datasets.classification_dataset import ClassificationDataset
from model.rescale import rescale18, rescale50
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from .utils import worker_init


class TCGAEstimator():
    def __init__(self, args: argparse.ArgumentParser, mode: str) -> None:
        self.args = args
        self.mode = mode

        # Can only have one of the following modes
        assert mode in ["train", "validation", "predict"], f"Invalid mode: {mode}"

        self.num_classes = args.num_classes
        self.drop_conv = args.drop_conv
        self.drop_fc = args.drop_fc
        self.model_type = args.model
        self.in_height = args.in_height
        self.in_width = args.in_width
        assert self.in_height == self.in_width, "Only square images are supported while training, predicting"
        self.image_size = args.in_height
        self.channels = args.channels
        self.batch_size = args.batch_size
        self.optimizer_type = args.optimizer
        self.inference = args.inference
        self.dataset_csv_path = args.dataset_csv_path
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.epochs = args.epochs
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir or args.log_dir
        self.ckpt_file = args.ckpt_file

        if mode in ["predict"] and args.ckpt_file is None:
            raise ValueError("Need to specify ckpt-file when mode is predict")

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.seed = args.seed
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor
        self.use_ddp = args.use_ddp
        self.data_dir = args.data_dir
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self._global_step = 0

        # Initialize model, optimizer and criterion
        self.model = self.init_model(args.model)
        self.optimizer = self.init_optim(self.model, args.model)
        self.criterion = nn.CrossEntropyLoss()
        if args.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            print("log dir not specified. Will not be logging")

    def __del__(self):
        if self.use_ddp and dist.is_initialized():
            self.cleanup_distributed_training()

    def setup_distributed_training(self) -> None:
        """
        Ensure distributed training is set up.
        Requires environment variables LOCAL_RANK and WORLD_SIZE.
        MASTER_ADDR and MASTER_PORT are optionally set.
        """
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12321'
        dist.init_process_group('nccl', init_method='env://')

    def cleanup_distributed_training(self) -> None:
        dist.destroy_process_group()

    def setup(self) -> None:
        """
        Load from a checkpoint if a pretrained model is defined
        """
        if self.mode in ["train", "predict"] and self.ckpt_file is not None:
            self.load_checkpoint(self.ckpt_file)

    def init_model(self, model_type: str) -> torch.nn.Module:
        """
        Create model with pretrained weights.
        """
        if model_type.startswith("rescale"):
            if model_type == "rescale18":
                model_class = rescale18
            elif model_type == "rescale50":
                model_class = rescale50
            else:
                raise Exception(f"Unsupported rescale:{model_type} type")
            model = model_class(num_classes=self.num_classes,
                                drop_conv=self.drop_conv,
                                drop_fc=self.drop_fc,
                                input_shapes=(self.in_height, self.in_width))
            if self.inference:
                model.eval()
        else:
            raise Exception("{model_type} not supported")

        if self.use_ddp:
            self.setup_distributed_training()
            torch.cuda.set_device(self.rank)
            model = model.to(self.rank)
            # initialize distributed data parallel (DDP)
            model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        else:
            model.cuda()
        return model

    def init_optim(self, model: torch.nn.Module, model_type: str) -> torch.optim:
        """
        Initialize optimizer based on the model type.
        """
        if model_type.startswith("rescale"):
            params_w_decay = []
            params_wo_decay = []
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if 'addbias' in name or '_scale' in name:
                        params_wo_decay.append(p)
                    else:
                        params_w_decay.append(p)
            if self.optimizer_type == 'adamw':
                optim = [
                    torch.optim.AdamW(params_wo_decay, lr=self.learning_rate, betas=(0.9, 0.997), weight_decay=0),
                    torch.optim.AdamW(
                        params_w_decay, lr=self.learning_rate, betas=(0.9, 0.997), weight_decay=self.weight_decay)
                ] if not self.inference else None
            elif self.optimizer_type == 'sgd':
                optim = [
                    torch.optim.SGD(params_wo_decay, lr=self.learning_rate, weight_decay=0, momentum=self.momentum),
                    torch.optim.SGD(
                        params_w_decay, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
                ] if not self.inference else None

        else:
            raise Exception("InvalidConfig: Optimizer could not be initialized")

        return optim

    def get_train_dataloader(self) -> DataLoader:
        """
        Return train dataloader.
        """
        transforms = [
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20, resample=PIL.Image.BILINEAR),
                T.ToTensor(),
        ]

        transform_train = T.Compose(transforms)
        mode = "train"

        if self.dataset_csv_path is not None:
            dataset = ClassificationDataset(data_csv_path=self.dataset_csv_path,
                                            data_dir=self.data_dir,
                                            subset=mode,
                                            transform=transform_train)
        else:
            raise Exception("Need a dataset csv to initialize the dataloader")

        sampler = DistributedSampler(dataset, seed=self.seed) if dist.is_initialized() else RandomSampler(dataset)
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  drop_last=True,
                                  num_workers=self.num_workers,
                                  worker_init_fn=worker_init,
                                  persistent_workers=self.num_workers > 0,
                                  prefetch_factor=self.prefetch_factor)
        return train_loader

    def get_test_dataloader(self, mode: str = None) -> DataLoader:
        """
        Return val or test dataloader.
        """
        assert mode in ["validation", "predict"], "Invalid mode for test dataloader"
        print(f"Mode for test dataloader is {mode}")

        transforms = [T.ToPILImage(), T.Resize((self.image_size, self.image_size)), T.ToTensor()]
        transform_test = T.Compose(transforms)

        if self.dataset_csv_path is not None:
            dataset = ClassificationDataset(data_csv_path=self.dataset_csv_path,
                                            data_dir=self.data_dir,
                                            subset=mode,
                                            transform=transform_test)
        else:
            raise Exception("Need a dataset csv to initialize the dataloader")

        sampler = DistributedSampler(dataset) if dist.is_initialized() else None

        test_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 drop_last=True,
                                 sampler=sampler,
                                 num_workers=self.num_workers,
                                 worker_init_fn=worker_init,
                                 persistent_workers=self.num_workers > 0,
                                 prefetch_factor=self.prefetch_factor)
        return test_loader

    def forward_pass(self, input_tensors: Tuple[torch.Tensor]) -> torch.Tensor:
        output = self.model(input_tensors)
        return output

    def backward_pass(self, loss: torch.Tensor) -> None:
        """
        Computes back gradient and optimizer step.
        """
        # Create optimizer List
        optim = self.optimizer if type(self.optimizer) == list else [self.optimizer]

        loss.backward()

        # Run Optimizer Step.
        for opt in optim:
            opt.step()

        # Zero the gradients.
        for opt in optim:
            opt.zero_grad()

    def train(self) -> None:
        """
        The main function which controls training.
        """
        # Initialize training
        self.setup()

        print("Running training")
        print(self.args)

        train_loader = self.get_train_dataloader()
        start_epoch = self._global_step // len(train_loader)

        print(f"start epoch: {start_epoch}")
        print(f"total epochs: {self.epochs}")

        # start training and save checkpoints after every epoch
        for epoch in tqdm(range(start_epoch, self.epochs)):
            self.save_checkpoint(epoch)

    def train_epoch(self, train_loader: DataLoader, epoch: int, sampler: DistributedSampler) -> Dict[str, Any]:
        """
        Train for one epoch.
        """
        self.model.train()

        if dist.is_initialized():
            sampler.set_epoch(epoch)
            dist.barrier()

        num_train_steps = len(train_loader)
        train_iter = iter(train_loader)
        print(f'Training Epoch: {epoch}')

        for i in range(num_train_steps):
            start_time = time.time()
            sample = next(train_iter)
            images, target, _ = sample
            images, target = images.cuda(), target.cuda()
            output = self.forward_pass(images)
            loss = self.criterion(output, target)
            self.backward_pass(loss)
            log_metrics = self.compute_log_metrics("train", output, target, loss)
            log_metrics["epoch"] = epoch
            end_time = time.time()
            log_metrics["time_per_step"] = end_time - start_time
            if self.rank == 0 and i % self.print_freq == 0:
                print(f"train - {self._global_step}", log_metrics)
            self._global_step += 1
        return log_metrics

    def validation(self, val_loader: DataLoader, ckpt_file: str) -> Dict[str, Any]:
        self.load_checkpoint(ckpt_file)
        mode = "validation"
        self.model.eval()
        with torch.no_grad():
            accuracy_metrics = []
            for sample in tqdm(val_loader):
                images, target, patient_ids = sample
                images, target = images.cuda(), target.cuda()
                output = self.forward_pass(images)
                loss = self.criterion(output, target)
                metrics = self.compute_log_metrics(mode, output, target, loss)
                metrics["patient_ids"] = patient_ids
                metrics["prediction"] = output
                metrics["target"] = target
                accuracy_metrics.append(metrics)
            val_metrics = self.aggregate_test_metrics(accuracy_metrics)
            print(f"validation - {self._global_step}", val_metrics)
            return val_metrics

    def evaluation(self, mode: str, ckpt_file: str) -> None:
        loader = self.get_test_dataloader(mode=mode)
        loader_iter = iter(loader)
        self.model.eval()

        all_pat_ids = []
        all_patch_predictions = np.zeros((0, self.num_classes))
        all_patch_labels = []

        with torch.no_grad():
            for _ in range(len(loader)):
                sample = next(loader_iter)
                images, target, pids = sample
                images, target = images.cuda(), target.cuda()
                all_pat_ids.extend(pids)
                all_patch_labels.extend(target.cpu())
                output = self.forward_pass(images)
                all_patch_predictions = np.concatenate((all_patch_predictions, output.cpu().detach().numpy()))
                predict = torch.argmax(output, dim=-1)
                for ii, patient_id in enumerate(pids):
                    pred_dict = {"pid": patient_id, "target": target[ii].item(), "predicted": predict[ii].item()}
                    print(f"predict - {self._global_step}", pred_dict)
            np.save(f"{self.log_dir}/all_pat_ids.npy", all_pat_ids)
            np.save(f"{self.log_dir}/all_patch_labels.npy", all_patch_labels)
            np.save(f"{self.log_dir}/all_scores.npy", all_patch_predictions)

    def compute_log_metrics(self,
                            mode: str,
                            prediction: torch.Tensor,
                            label: torch.Tensor,
                            loss: torch.Tensor) -> Dict[str, Any]:
        log_metrics = {}
        log_metrics["loss"] = loss.item()
        prediction = torch.argmax(prediction, dim=-1)
        log_metrics["accuracy"] = self.get_patch_accuracy(prediction, label)
        return log_metrics

    def get_patch_accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        return (preds == labels).sum() / np.prod(preds.shape)

    def aggregate_test_metrics(self, accuracy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        all_labels, all_scores, all_patient_ids = [], np.zeros((0, self.num_classes)), []
        loss_sum = 0
        acc_sum = 0
        for metrics in accuracy_metrics:
            all_labels.extend(metrics["target"])
            all_patient_ids.extend(metrics["patient_ids"])
            all_scores = np.concatenate((all_scores, metrics["prediction"].cpu().detach().numpy()))
            loss_sum += metrics.get("loss", 0)
            acc_sum += metrics.get("accuracy", 0)
        score_tensor = torch.from_numpy(all_scores)
        softmax = nn.Softmax(dim=-1)
        scores = softmax(score_tensor)
        auc_score = self.get_patient_auc(scores, all_labels, all_patient_ids)
        val_metrics = {
            "loss": loss_sum / len(accuracy_metrics),
            "accuracy": acc_sum / len(accuracy_metrics),
            "auc": auc_score
        }
        return val_metrics

    def get_patient_auc(self, probs: torch.Tensor, labels: list, pat_ids: List) -> float:
        default = {"predictions": [], "target": None}
        patient_predictions = {}
        for ii, (pat_id, target) in enumerate(zip(pat_ids, labels)):
            if pat_id not in patient_predictions:
                patient_predictions[pat_id] = copy.deepcopy(default)
            prediction = torch.argmax(probs[ii], dim=-1)
            patient_predictions[pat_id]["predictions"].append(prediction.item())
            patient_predictions[pat_id]["target"] = target.item()

        patient_targets = []
        patient_scores = np.zeros((len(patient_predictions), self.num_classes))
        for ii, (_, info_dict) in enumerate(patient_predictions.items()):
            patient_targets.append(info_dict["target"])
            # Calculate patient-level class probability distribution
            for class_num in range(self.num_classes):
                patient_scores[ii, class_num] = sum(np.array(info_dict["predictions"]) == class_num)
            patient_scores[ii] = patient_scores[ii] / sum(patient_scores[ii])
        # Compute one vs. rest auroc
        auc_score = roc_auc_score(patient_targets, np.array(patient_scores), average='macro', multi_class='ovr')
        return auc_score

    def get_state_dict(self, ckpt_file) -> Dict:
        '''
        Load checkpoint from file.
        '''
        if ckpt_file is not None:
            print(f"Restoring weights from {ckpt_file}")
            state = torch.load(ckpt_file)
            print("State dict loaded successfully")
            return state

        return None

    def load_checkpoint(self, ckpt_file) -> bool:
        """
        Load model state from an existing checkpoint.
        """
        state = self.get_state_dict(ckpt_file)
        if state is None:
            print("Could not load from checkpoint file")
            return False

        if 'module' in list(state['state_dict'].keys())[0]:
            updated_state_dict = collections.OrderedDict()
            for key, val in state['state_dict'].items():
                updated_state_dict[key.replace('module.', '')] = val
            state['state_dict'] = updated_state_dict

        model_keys = set([key for key, _ in self.model.named_parameters()])
        dict_keys = set(state['state_dict'].keys())
        if model_keys.intersection(dict_keys) != model_keys:
            missing_keys = '\n\t' + '\n\t'.join(model_keys - model_keys.intersection(dict_keys))
            print(
                f'Warning: the following keys appear in the model but not in the checkpoint:{missing_keys}')
        if dict_keys.intersection(model_keys) != dict_keys:
            missing_keys = '\n\t' + '\n\t'.join(dict_keys - dict_keys.intersection(model_keys))
            print(
                f'Warning: the following keys appear in the checkpoint but not in the model:{missing_keys}')

        self.model.load_state_dict(state['state_dict'])
        print("Checkpoint loaded successfully!!")
        self._global_step = state.get('global_step', 0)

        return True

    def get_model_filepath(self, is_best_ckpt: bool = False) -> Tuple[str]:
        '''
        Helper function to build the filepath of a model for saving and restoring:
        '''
        if self.ckpt_dir is None:
            return None, None
        file_path = os.path.join(self.ckpt_dir, "checkpoints")
        if is_best_ckpt:
            name = os.path.join(file_path, 'model-best-{}.ckpt'.format(self._global_step))
            checkpoint_file_path = os.path.join(file_path, "checkpoint_best")
        else:
            name = os.path.join(file_path, 'model-{}.ckpt'.format(self._global_step))
            checkpoint_file_path = os.path.join(file_path, "checkpoint")
        return name, checkpoint_file_path

    def save_checkpoint(self, epoch: int, is_best_ckpt: bool = False) -> None:
        '''
        Save the model to file.
        '''
        if self.rank != 0:
            # Save only for rank0.
            return

        current_file_path, checkpoint_file_path = self.get_model_filepath(is_best_ckpt)
        if current_file_path is None:
            print("Not saving checkpoint. Checkpoint dir is None")
            return

        optim = self.optimizer if type(self.optimizer) == list else [self.optimizer]
        state_dict = {
            'global_step': self._global_step,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in optim],
            'epoch': epoch
        }

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]): vals[1].replace(' ', '')})
        except Exception as exc:
            print(f'Exception {exc} was thrown')

        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                if key != self._global_step:  # overwrite past checkpoints with same name
                    _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))
