
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
from typing import Dict, List, Set, Tuple

import pandas as pd

from . import dataframe_utils as df_utils
from . import summarize


def validate_args(args):
    # sanity checks regarding the various folders
    assert os.path.exists(args.labels_folder), 'Labels folder does not exist!'
    for patient_id_folder in args.patient_id_folders:
        assert os.path.exists(patient_id_folder), f'Patient Id Folder {patient_id_folder} does not exist'

    # sanity checks for test and valid splits
    err_msg = 'Valid split and test split must be between 0 and 1'
    assert 0. <= args.valid_split <= 1. and 0. <= args.test_split <= 1., err_msg
    err_msg = 'Each fold must have a unique test set, no patient id can exist in more than one test set'
    assert args.test_split * args.num_folds <= 1, err_msg


def validate_labels(labels_df, print_summary=False):
    err_msg = 'Duplicate Patient Ids found in Label files!'
    assert len(labels_df) == len(set(labels_df[df_utils.LabelCols.patient_id])), err_msg
    if print_summary:
        summarize.summarize_labels(labels_df)


def validate_folders(folder_to_patient_ids: Dict[str, List[str]], print_summary=False):
    # verifying there aren't duplicate patient ids in different patient id folders
    duplicate_patient_ids = set()

    # no duplicates within a patient id folder
    for folder, patient_ids in folder_to_patient_ids.items():
        assert len(set(patient_ids)) == len(patient_ids), f'Duplicate patient ids in {folder}'

    # no duplicates across patient id folders
    for i, (folder_1, patient_ids_1) in enumerate(folder_to_patient_ids.items()):
        for j, (folder_2, patient_ids_2) in enumerate(folder_to_patient_ids.items()):
            if i == j:
                continue
            duplicate_patient_ids = set(patient_ids_1).intersection(set(patient_ids_2))
            assert duplicate_patient_ids == set(
            ), f'Duplicate patient ids {duplicate_patient_ids} found in {folder_1} and {folder_2}'

    if print_summary:
        summarize.summarize_folders(folder_to_patient_ids)


def validate_patches(folder_to_patient_ids_to_patches: Dict[str, Dict[str, str]],
                     patched_patient_ids: Set[str],
                     labels_df: pd.DataFrame,
                     print_summary: bool = False):
    patient_id_to_slide_id = {}
    for folder, patient_ids_to_patches in folder_to_patient_ids_to_patches.items():
        for patient_id, patches in patient_ids_to_patches.items():
            slide_ids = set()
            for patch in patches:
                slide_ids.add(patch.split('_')[0])
            err_msg = f'Multiple slide ids {slide_ids} found in {folder}/{patient_id}'
            assert len(slide_ids) == 1, err_msg
            patient_id_to_slide_id[patient_id] = slide_ids.pop()

    with open('patient_id_to_slide_id.txt', 'w') as f:
        for patient_id, slide_id in patient_id_to_slide_id.items():
            f.write(f'{patient_id} {slide_id}\n')

    # patient ids which do not have a folder of patches
    not_patched_patient_ids = set()
    for patient_id in labels_df[df_utils.LabelCols.patient_id]:
        if patient_id not in patched_patient_ids:
            not_patched_patient_ids.add(patient_id)

    if print_summary:
        summarize.summarize_patches(folder_to_patient_ids_to_patches, patched_patient_ids, not_patched_patient_ids)

    if len(not_patched_patient_ids) > 0:
        with open('not_patched_patient_ids.txt', 'w') as not_patched_patient_ids_file:
            for patient_id in not_patched_patient_ids:
                not_patched_patient_ids_file.write(f'{patient_id}\n')
        # raise RuntimeError('Not Patched Patient IDs found.  Written to not_patched_patient_ids.txt')


def validate_split_indices(num_patient_ids: int, split_indices: List[int], print_summary=False):
    all_test_indices = set()
    all_test_indices_length = 0
    for i, (train_indices, test_indices) in enumerate(split_indices):
        err_msg = f'Missing some patient ids in fold {i}'
        assert len(train_indices) + len(test_indices) == num_patient_ids, err_msg
        all_test_indices = all_test_indices.union(test_indices)
        all_test_indices_length += len(test_indices)
    assert len(all_test_indices) == all_test_indices_length, 'Found duplicates in test indices across folds'

    if print_summary:
        summarize.summarize_split_indices(num_patient_ids, split_indices)


def validate_dfs(train_df,
                 valid_df,
                 test_df,
                 labels_df,
                 num_patient_ids,
                 fold_index,
                 patches_per_patient_id=None,
                 print_summary=False):
    # check that the dataframes are not missing any patient ids
    train_patient_ids = set(train_df[df_utils.NonBinnedCsvCols.patient_id])
    valid_patient_ids = set(valid_df[df_utils.NonBinnedCsvCols.patient_id])
    test_patient_ids = set(test_df[df_utils.NonBinnedCsvCols.patient_id])

    # Train and validation should not have duplicates with the test set
    train_and_test_intersect = train_patient_ids.intersection(test_patient_ids)
    valid_and_test_intersect = valid_patient_ids.intersection(test_patient_ids)
    assert train_and_test_intersect == set(), f'Train and test have duplicate patient ids {train_and_test_intersect}'
    assert valid_and_test_intersect == set(), f'Valid and test have duplicate patient ids {valid_and_test_intersect}'

    err_msg = 'Number of patient ids in dataframes is not equal to total number of patient ids'
    assert len(train_patient_ids.union(valid_patient_ids)) + len(test_patient_ids) == num_patient_ids

    # check that sampling occurred
    patient_ids = [train_patient_ids, valid_patient_ids, test_patient_ids]
    dfs = [train_df, valid_df, test_df]
    for split_patient_ids, df, _ in zip(patient_ids, dfs, ['train', 'valid', 'test']):
        deceased_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 1]
        survived_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 0]
        assert len(deceased_df) > 0, 'Must have some deceased patient examples in dataset'
        assert len(survived_df) > 0, 'Must have some survived patient examples in dataset'
        assert len(survived_df) + len(deceased_df) == len(df), 'Some examples have an invalid label'
        if patches_per_patient_id is not None:
            for patient_id in split_patient_ids:
                patient_id_df = df[df[df_utils.NonBinnedCsvCols.patient_id] == patient_id]
                unique_image_paths = patient_id_df[df_utils.NonBinnedCsvCols.image_path].unique()
                err_msg = f'Too many samples {len(unique_image_paths)} from {patient_id}'
                assert len(unique_image_paths) <= patches_per_patient_id, err_msg

    if print_summary:
        summarize.summarize_dfs(train_df, valid_df, test_df, labels_df, fold_index)


def validate_upsampled_dfs(upsampled_dfs: List[Tuple[pd.DataFrame, pd.DataFrame, str]], print_summary=False):
    for df, upsampled_df, df_name in upsampled_dfs:
        deceased_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 1]
        survived_df = df[df[df_utils.NonBinnedCsvCols.status_label] == 0]

        upsampled_deceased_df = upsampled_df[upsampled_df[df_utils.NonBinnedCsvCols.status_label] == 1]
        upsampled_survived_df = upsampled_df[upsampled_df[df_utils.NonBinnedCsvCols.status_label] == 0]

        assert len(survived_df) == len(upsampled_survived_df), "Upsampling of survived examples should not occur!"
        assert len(deceased_df) < len(upsampled_deceased_df), "No upsampling occurred!"
        err_msg = "Number of deceased and survived examples in upsampled dataframe should be the same"
        assert len(upsampled_deceased_df) == len(upsampled_survived_df), err_msg

        if print_summary:
            summarize.summarize_upsampled_df(df, upsampled_df, df_name)


def validate_binned_outputs(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, print_summary=False):
    dfs = [train_df, valid_df, test_df]
    for df in dfs:
        assert len(df[df_utils.FinalCsvCols.label].unique()) <= 3, 'At most 3 categories for labels after binning'

    if print_summary:
        summarize.summarize_binned_outputs(train_df, valid_df, test_df)


def validate_columns(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    dfs = [train_df, valid_df, test_df]
    for df in dfs:
        assert list(df.columns) == df_utils.FinalCsvCols.cols
