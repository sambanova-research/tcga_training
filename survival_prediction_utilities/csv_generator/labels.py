
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

import pandas as pd

from . import dataframe_utils as df_utils


def format_labels_df(labels_df: pd.DataFrame, label_file_path: str):
    # rename the columns to the required column names
    for label_col, orig_label_cols in zip(df_utils.LabelCols.cols, df_utils.OrigLabelCols.cols):
        did_rename = False
        for orig_label_col in orig_label_cols:
            if orig_label_col in labels_df.columns:
                labels_df[label_col] = labels_df[orig_label_col]
                did_rename = True
                break
        if not did_rename:
            raise ValueError(f'Must have one of the columns {orig_label_cols} in label file {label_file_path}')

    # delete the unnnecessary columns
    extra_columns = list(set(labels_df.columns).difference(set(df_utils.LabelCols.cols)))
    labels_df = labels_df.drop(extra_columns, axis=1)
    return labels_df


def get_labels(labels_folder: str) -> pd.DataFrame:
    """ Creates a single, combined dataframe with all the survival prediction labels

    Args:
        labels_folder: Folder containing all the label csv/tsv files provided by Prof Kun.

    Returns:
        The survival prediction labels dataframe
    """
    labels = []
    for label_file in os.listdir(labels_folder):
        assert label_file.endswith('.tsv') or label_file.endswith('.csv'), 'Labels file must be a tsv or csv file'
        label_file_path = os.path.join(labels_folder, label_file)
        sep = '\t' if label_file.endswith('.tsv') else ','
        labels_df = pd.read_csv(label_file_path, sep=sep, na_filter=False)
        # tsvs and csvs do not have same format, create unified format
        labels_df = format_labels_df(labels_df, label_file_path)
        labels.append(labels_df)
    combined_labels_df = pd.concat(labels, ignore_index=True)
    survival_event_converter = lambda s: int(s.split(':')[0]) if isinstance(s, str) else s
    combined_labels_df[df_utils.LabelCols.survival_event] = combined_labels_df[df_utils.LabelCols.survival_event].apply(
        survival_event_converter)

    return combined_labels_df
