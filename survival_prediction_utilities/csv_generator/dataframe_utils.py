
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


import itertools
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import survival.mtlr_cnn as mtlr
import torch

DF_SPLIT_NAMES = ['train', 'valid', 'test']
NUM_OUTPUT_BINS = 2
SUBSET_TRAIN = 'train'
SUBSET_VALID = 'validation'
SUBSET_TEST = 'predict'


class FinalCsvCols:
    """ Columns that the final train / validation / test dataframes that we generate will have """
    image_path = 'image_path'
    label = 'label'
    subset = 'subset'
    patient_ids = 'metadata'
    duration = 'duration'
    status_label = 'status_label'
    duration_label = 'duration_label'
    cols = [image_path, label, subset, patient_ids, status_label, duration_label]


class LabelCols:
    """ Columns of the Label dataframes that we generate"""
    patient_id = 'patient_id'
    survival_duration = 'survival_duration'
    survival_event = 'survival_event'
    cols = [patient_id, survival_duration, survival_event]


class NonBinnedCsvCols:
    """ Columns of the dataframes we generate before the final label binning is applied """
    image_path = 'image_path'
    status_label = 'status_label'
    duration_label = 'duration_label'
    subset = 'subset'
    patient_id = 'metadata'
    cols = [image_path, status_label, duration_label, subset, patient_id]


class OrigLabelCols:
    """ Columns that the Label files that are inputs to this program must have """
    patient_id = ('Patient ID', 'id')
    survival_durations = ('Overall Survival (Months)', 'months')
    survival_events = ('Overall Survival Status', 'event')
    cols = [patient_id, survival_durations, survival_events]


def bin_outputs(train_df: pd.DataFrame, upsampled_train_df: pd.DataFrame, valid_df: pd.DataFrame,
                test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Bins the survival duration and survival status labels into vectors to make the training problem easier.
    Example 1:
        Let's assume NUM_OUTPUT_BINS = 2, survival duration is longer than middle
            ([max(survival duration) - min(survival_duration)] / 2), and survival status is Deceased or Survived.
        Binned label:  [0, 1]

    Example 2:
        Let's assume NUM_OUTPUT_BINS = 2, survival duration is shorter than middle and survival status is Deceased.
        Binned label:  [1, 0]

    Example 3:
        Let's assume NUM_OUTPUT_BINS = 2, survival duration is shorter than middle and survival status is Survived.
        Binned label:  [1, 1]

    Args:
        train_df: The training dataframe
        valid_df: The validation dataframe
        test_df:  The test dataframe
        upsampled_train_df: The upsampled training dataframe

    Returns:
        The upsampled training dataframe, the validation dataframe and the test dataframe.
    """
    split_labels = {}

    dfs = (upsampled_train_df, valid_df, test_df, train_df)
    df_names = DF_SPLIT_NAMES + ['old_train']

    # extract just the labels for each dataframe
    for split_name, df in zip(df_names, dfs):
        tensor_labels = []
        for _, row in df.iterrows():
            status_label = row[NonBinnedCsvCols.status_label]
            duration_label = row[NonBinnedCsvCols.duration_label]
            labels = torch.tensor([duration_label, status_label])
            tensor_labels.append(labels)
        split_labels[split_name] = torch.stack(tensor_labels, dim=0)

    # bin the labels and add them back to the dataframes
    dfs = (upsampled_train_df, valid_df, test_df)
    final_dfs = []
    for split_name, df in zip(df_names, dfs):
        binned_labels = mtlr.binSurvival(split_labels['old_train'], split_labels[split_name], NUM_OUTPUT_BINS)
        df[[f'y_bin_{num}' for num in range(NUM_OUTPUT_BINS)]] = binned_labels

        # Convert binned labels to single number so that we can train on RDU according to this:
        # [1, 0] -> 0;  [0, 1] -> 1;  [1, 1] -> 2
        label_column = df.apply(lambda row: 0 if row.y_bin_0 == 1 and row.y_bin_1 == 0 else row.y_bin_0 + row.y_bin_1,
                                axis=1)
        label_column = label_column.astype(int)
        df.insert(1, FinalCsvCols.label, label_column)
        final_dfs.append(df)
    return tuple(final_dfs)


def get_split_indices(num_patient_ids: int, num_folds: int, valid_split: float,
                      test_split: float) -> List[List[List[int]]]:
    """ Get the indices of the patient ids for each training, validation, and test split in each fold.

    Args:
        num_patient_ids: Total number of patient ids
        num_folds: Number of folds of the dataset to create
        valid_split:  The validation split as specified by the user.  The validation split is a fraction of the
            (whole dataset - test data) rather than a fraction of the whole dataset.
        test_split:  The test split as specified by the user.  The test split is a fraction of the whole dataset

    Returns:
        The indices
    """
    orig_indices = np.array_split(range(num_patient_ids), int(1 / test_split))
    split_indices = []
    for i in range(num_folds):
        # create a copy
        indices = [array for array in orig_indices]
        test_indices = indices.pop(i).tolist()

        random.shuffle(indices)
        # flatten the remaining indices
        train_indices = list(itertools.chain.from_iterable(indices))
        split_indices.append([train_indices, test_indices])
    return split_indices


def remove_extra_columns(dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Deletes any intermediate columns that were required when generating the DataFrame but are not required in the
    final version of the dataframe.
    Args:
        dfs: Train, validation and test dataframes
    Returns:
        The train, validation and test dataframes after removing the unnecessary columns
    """
    final_dfs = []
    for df in dfs:
        unnecessary_columns = set(df.columns).difference(FinalCsvCols.cols)
        df = df.drop(labels=unnecessary_columns, axis=1)
        # reorder columns as specified by FinalCsvCols
        df = df[FinalCsvCols.cols]
        final_dfs.append(df)
    return tuple(final_dfs)


def upsample_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """ Upsamples the number of deceased examples in the training dataframe so that the number of deceased examples
    and the number of survived examples are equal.

    Args:
        train_df: The training dataframe

    Returns:
        The upsampled training dataframe
    """
    deceased_df = train_df[train_df[NonBinnedCsvCols.status_label] == 1]
    survived_df = train_df[train_df[NonBinnedCsvCols.status_label] == 0]
    upsampled_deceased_df = deceased_df.sample(n=len(survived_df), replace=True, ignore_index=True)
    upsampled_train_df = pd.concat((survived_df, upsampled_deceased_df), ignore_index=True)

    # shuffle dataframe because after the concat in the previous line, all the survived examples are in the first half
    # and all the deceased examples are in the second half
    upsampled_train_df = upsampled_train_df.sample(frac=1, ignore_index=True)
    return upsampled_train_df
