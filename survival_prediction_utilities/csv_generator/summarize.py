
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


import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from . import dataframe_utils as df_utils

HEADER = f"\n{'=' * 50}\n\t%s\n{'=' * 50}"


def summarize_labels(labels_df: pd.DataFrame):
    survival_duration_col = labels_df[df_utils.LabelCols.survival_duration]
    survival_event_col = labels_df[df_utils.LabelCols.survival_event]
    print(HEADER % 'LABELS')
    print(labels_df.head())
    print(f'Number of labels: {len(labels_df)}')
    num_survived_examples = len(labels_df[survival_event_col == 0])
    num_deceased_examples = len(labels_df[survival_event_col == 1])
    print(f'Number of survived examples: {num_survived_examples}')
    print(f'Number of deceased examples: {num_deceased_examples}')
    print('Survival duration statistics:')
    print(survival_duration_col.describe(), '\n')
    random_predictions = np.random.choice([0, 1], size=len(labels_df), p=[.5, .5])
    random_ci = concordance_index(survival_duration_col, random_predictions, survival_event_col)
    print(f'Random Guessing Produced a Concordance Index of {random_ci}')


def summarize_folders(folder_to_patient_ids: Dict[str, List[str]]):
    print(HEADER % 'PROCESSED PATIENT IDs ')
    num_patient_ids = 0
    for folder, patient_ids in folder_to_patient_ids.items():
        print(f'Number of patient IDs in {folder} is {len(patient_ids)}')
        num_patient_ids += len(patient_ids)
    print(f'Total number of patient IDs with patch subfolders is {num_patient_ids}\n')


def summarize_patches(folder_to_patient_ids_to_patches, patched_patient_ids: Set[str],
                      not_patched_patient_ids: Set[str]):
    print(HEADER % 'PATCHED PATIENT IDS')
    print(f'Number of patched patient ids {len(patched_patient_ids)}')
    print(f'Number of not patched patient ids {len(not_patched_patient_ids)}\n')

    num_patches_per_patients = []
    patient_id_paths = []
    for folder, patient_ids_to_patches in folder_to_patient_ids_to_patches.items():
        for patient_id, patches in patient_ids_to_patches.items():
            num_patches_per_patients.append(len(patches))
            patient_id_paths.append(os.path.join(folder, patient_id))
    num_patches_per_patients = np.array(num_patches_per_patients)
    max_index = np.argmax(num_patches_per_patients)
    min_index = np.argmin(num_patches_per_patients)
    max_patches = num_patches_per_patients[max_index]
    min_patches = num_patches_per_patients[min_index]
    max_path = patient_id_paths[max_index]
    min_path = patient_id_paths[min_index]

    print(f'{max_path} has the most number of patches: {max_patches}')
    print(f'{min_path} has the least number of patches: {min_patches}')
    avg_patches = round(sum(num_patches_per_patients) / len(num_patches_per_patients))
    print(f'Avg number of patches per patient id {avg_patches}')


def summarize_split_indices(num_patient_ids: int, split_indices: List[List[str]]):
    print(HEADER % 'PATIENT ID SPLITS')
    for i, (train_indices, test_indices) in enumerate(split_indices):
        train_split = round(len(train_indices) / num_patient_ids, 3)
        test_split = round(len(test_indices) / num_patient_ids, 3)
        print(f'Fold {i} patient id splits are:')
        print(f'\tPatientIds: train - {len(train_indices)}, test - {len(test_indices)}')
        print(f'\tFraction: train - {train_split}, test - {test_split}')


def summarize_dfs(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, label_df: pd.DataFrame,
                  fold_index: int):
    print(HEADER % f'FOLD {fold_index} Data Frames')
    num_patches = len(train_df) + len(valid_df) + len(test_df)
    train_split = round(len(train_df) / num_patches, 3)
    valid_split = round(len(valid_df) / num_patches, 3)
    test_split = round(len(test_df) / num_patches, 3)

    print('Patch splits are:')
    print(f'\tPatches: train - {len(train_df)}, valid - {len(valid_df)}, test - {len(test_df)}')
    print(f'\tFraction: train - {train_split}, valid - {valid_split}, test - {test_split}\n')

    print('Class division is:')
    dfs = [train_df, valid_df, test_df]
    for name, df in zip(df_utils.DF_SPLIT_NAMES, dfs):
        deceased_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 1]
        survived_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 0]
        deceased_patient_ids = deceased_df[df_utils.NonBinnedCsvCols.patient_id].unique()
        survived_patient_ids = survived_df[df_utils.NonBinnedCsvCols.patient_id].unique()
        deceased_labels_df = label_df.loc[label_df[df_utils.LabelCols.patient_id].isin(deceased_patient_ids)]
        survived_labels_df = label_df.loc[label_df[df_utils.LabelCols.patient_id].isin(survived_patient_ids)]
        print(f'\t{name} division is {len(survived_labels_df)} survived and {len(deceased_labels_df)} deceased')


def summarize_upsampled_df(df: pd.DataFrame, upsampled_df: pd.DataFrame, df_name: str):
    deceased_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 1]
    survived_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 0]
    print(f'\nBefore upsampling {df_name} total number of examples is {len(df)}, \
          {len(deceased_df)} deceased examples and {len(survived_df)} survived examples')
    upsampled_deceased_df = upsampled_df[upsampled_df[df_utils.NonBinnedCsvCols.status_label] == 1]
    upsampled_survived_df = upsampled_df[upsampled_df[df_utils.NonBinnedCsvCols.status_label] == 0]
    print(f'After upsampling {df_name} total number of examples is {len(upsampled_df)}, \
          {len(upsampled_deceased_df)} deceased examples and {len(upsampled_survived_df)} survived examples')


def summarize_binned_outputs(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    print('\nClass division after binning is:')
    dfs = [train_df, valid_df, test_df]
    for name, df in zip(df_utils.DF_SPLIT_NAMES, dfs):
        class_0 = df[df[df_utils.FinalCsvCols.label] == 0]
        class_1 = df[df[df_utils.FinalCsvCols.label] == 1]
        class_2 = df[df[df_utils.FinalCsvCols.label] == 2]
        print(f'\t{name} division is {len(class_0)} class_0, {len(class_1)} class_1, {len(class_2)} class_2')
        random_guesses_correct = ((len(class_0) + len(class_1)) / 2) + len(class_2)
        total_samples = len(class_0) + len(class_1) + len(class_2)
        random_guessing_accuracy = round(random_guesses_correct / total_samples, 3)
        print(f'\tRandom guessing would get about {random_guessing_accuracy} accuracy')
