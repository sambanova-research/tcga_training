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
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

NUM_CLASSES = 3


def get_patch_accuracy(preds, labels):
    return (preds == labels).sum() / np.prod(preds.shape)


def evaluate_metrics(preds, labels):
    truepos = []
    falsepos = []
    trueneg = []
    falseneg = []
    for class_id in range(NUM_CLASSES):
        truepos.append(((preds == class_id) * (labels == class_id)).sum())
        falsepos.append(((preds == class_id) * (labels != class_id)).sum())
        trueneg.append(((preds != class_id) * (labels != class_id)).sum())
        falseneg.append(((preds != class_id) * (labels == class_id)).sum())
    return truepos, falsepos, trueneg, falseneg


def get_precision(truepos, falsepos, trueneg, falseneg):
    denomenator = truepos + falsepos
    if denomenator == 0:
        return 0
    return float(truepos) / float(denomenator)


def get_recall(truepos, falsepos, trueneg, falseneg):
    denomenator = truepos + falseneg
    if denomenator == 0:
        return 0
    return float(truepos) / float(denomenator)


def get_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def get_micro_stats(tp, fp, tn, fn, mode="patient"):
    results = []
    results.append(f"Micro scores at {mode} level")
    all_precision = []
    all_recall = []
    all_f1 = []
    for i in range(NUM_CLASSES):
        precision = get_precision(tp[i], fp[i], tn[i], fn[i])
        recall = get_recall(tp[i], fp[i], tn[i], fn[i])
        f1 = get_f1_score(precision, recall)
        results.append(f"Class id: {i}")
        results.append(f"precision: {precision} recall: {recall} f1: {f1}")
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    return all_precision, all_recall, all_f1, results


def get_patient_metrics(scores, labels, pat_ids, log_dir=None):
    results = []
    default = {"predictions": [], "target": None}
    patient_predictions = {}
    for ii, (pat_id, target) in enumerate(zip(pat_ids, labels)):
        if pat_id not in patient_predictions:
            patient_predictions[pat_id] = copy.deepcopy(default)
        prediction = np.argmax(scores[ii], axis=-1)
        patient_predictions[pat_id]["predictions"].append(prediction.item())
        patient_predictions[pat_id]["target"] = target.item()

    patient_targets = []
    patient_scores = np.zeros((len(patient_predictions), NUM_CLASSES))
    for ii, (_, info_dict) in enumerate(patient_predictions.items()):
        patient_targets.append(info_dict["target"])

        # Calculate patient-level class probability distribution
        for class_num in range(NUM_CLASSES):
            patient_scores[ii, class_num] = sum(np.array(info_dict["predictions"]) == class_num)
        patient_scores[ii] = patient_scores[ii] / sum(patient_scores[ii])

    patient_acc = sum(np.array(patient_targets) == np.argmax(patient_scores, axis=-1)) / len(patient_targets)

    # Compute one vs. rest auroc
    auc_pat = roc_auc_score(patient_targets, np.array(patient_scores), average='macro', multi_class='ovr')

    patient_scores = np.argmax(patient_scores, axis=-1)
    patient_targets = np.array(patient_targets)
    tp, fp, tn, fn = evaluate_metrics(patient_scores, patient_targets)
    all_precision, all_recall, all_f1, micro_results = get_micro_stats(tp, fp, tn, fn, mode="patient")

    results += micro_results
    results.append("Macro scores")
    results.append(f"Overall patient precision: {np.mean(all_precision)}")
    results.append(f"Overall patient recall: {np.mean(all_recall)}")
    results.append(f"Overall patient F1: {np.mean(all_f1)}")
    results.append(f"Patient AUC: {auc_pat}")
    results.append(f"Patient accuracy: {patient_acc}")
    return results


def write_log(results):
    with open(f"{log_dir}/results.txt", "a") as f:
        [f.write(result + "\n") for result in results]


def parse_args():
    parser = argparse.ArgumentParser(description='TCGA Metric evaluation')
    parser.add_argument('--artifact-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, default="output")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    artifact_dir = args.artifact_dir
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    labels = np.load(f"{artifact_dir}/all_patch_labels.npy")
    pat_ids = np.load(f"{artifact_dir}/all_pat_ids.npy")
    scores = np.load(f"{artifact_dir}/all_scores.npy")
    assert len(labels) == len(scores) == len(pat_ids)

    # convert raw scores to softmax scores
    score_tensor = torch.from_numpy(scores)
    softmax = nn.Softmax(dim=-1)
    scores = softmax(score_tensor)
    scores = scores.cpu().detach().numpy()
    preds = np.argmax(scores, axis=-1)

    results = []
    patch_acc = get_patch_accuracy(preds, labels)
    results.append(f"patch accuracy: {patch_acc}")
    tp, fp, tn, fn = evaluate_metrics(preds, labels)
    all_precision, all_recall, all_f1, micro_results = get_micro_stats(tp, fp, tn, fn, mode="patch")
    results += micro_results
    results.append("Macro scores")
    results.append(f"Overall patch precision: {np.mean(all_precision)}")
    results.append(f"Overall patch recall: {np.mean(all_recall)}")
    results.append(f"Overall patch F1: {np.mean(all_f1)}")

    patch_auc = roc_auc_score(labels, np.array(scores), average='macro', multi_class='ovr')
    results.append(f"patch auc: {patch_auc}")
    results += get_patient_metrics(scores, labels, pat_ids, log_dir=log_dir)

    write_log(results)
