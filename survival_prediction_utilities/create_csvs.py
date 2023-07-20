
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
import os
from typing import Dict, List, Optional, Set, Tuple

import csv_generator.dataframe_utils as df_utils
import csv_generator.labels as labels
import csv_generator.patched_patient_folders as patched_patient_folders
import csv_generator.utils as utils
import csv_generator.validate as validate
import numpy as np
import pandas as pd
from csv_generator.profiler import Profiler
from sklearn.model_selection import train_test_split

# constants
PROFILER = Profiler()
SEED = 0


def get_parser() -> argparse.ArgumentParser:
    """ Creates the command line argument parser """
    parser = argparse.ArgumentParser('Create training csvs that will be used to train our survival prediction models')
    parser.add_argument('--do-random-sampling',
                        action='store_true',
                        default=False,
                        help='Samples the patient ids by number of non-background pixels')
    parser.add_argument('--labels-folder',
                        type=str,
                        required=True,
                        metavar='LABELS_FOLDER',
                        help='Folder containing the survival labels provided by Professor Kun in tsv format.')
    parser.add_argument('--num-folds', type=int, default=10, metavar='NUM_FOLDS', help='Number of folds')
    parser.add_argument('--num-output-bins',
                        type=int,
                        default=None,
                        help='Number of bins to bin the output into.  If None, does not do any binning')
    parser.add_argument('--output-folder',
                        type=str,
                        required=True,
                        metavar='OUTPUT_FOLDER',
                        help='Folder in which to save the train, valid and test csvs.')
    parser.add_argument('--patches-per-patient-id',
                        type=int,
                        default=None,
                        help='Number of patches to sample per patient id.  If None, uses all patients')
    parser.add_argument('--patient-id-folders',
                        type=str,
                        nargs='+',
                        required=True,
                        metavar='PATIENT_ID_FOLDERS',
                        help='Folders containing the processed WSI patches for each patient id.')
    parser.add_argument('--print-summary',
                        action='store_true',
                        default=False,
                        help='Prints a summary about the labels and processed patient ids')
    parser.add_argument('--upsample-valid',
                        action='store_true',
                        default=False,
                        help='Upsamples the number of Deceased patients in the validation set')
    parser.add_argument('--rm-existing-output-dir',
                        action='store_true',
                        default=False,
                        help='Deletes the existing output directory if it exists')
    parser.add_argument('--test-split', type=float, default=0.2, metavar='TEST_SPLIT', help='Test Split')
    parser.add_argument('--valid-split', type=float, default=0.2, metavar='VALID_SPLIT', help='Validation Split')
    return parser


def sample_patches_incrementally(patient_id_df: pd.DataFrame, patches_per_patient_id: int):
    patch_rows = []
    patch_scores = []
    for _, row in patient_id_df.iterrows():
        patch_rows.append(row)
        patch_path = row['image_path']
        indices_lst = patch_path.split('.')[0].split('_')
        patch_score = int(indices_lst[-1]) * int(indices_lst[-2])
        patch_scores.append(patch_score)
    indices = np.argsort(np.array(patch_scores))
    incremental_indices = [indices[i] for i in np.linspace(0, len(patient_id_df), num=200).astype(int)]
    sampled_patch_rows = [patch_rows[i] for i in incremental_indices]
    return pd.DataFrame(sampled_patch_rows)


def create_df(labels_df: pd.DataFrame,
              folder_to_patient_ids_to_patches: Dict[str, Dict[str, str]],
              patient_ids: List[str],
              patient_indices: List[str],
              subset: str,
              patches_per_patient_id: Optional[int] = None,
              do_random_sampling: bool = False) -> pd.DataFrame:
    """ Creates a dataframe out of the specified subset of patient ids and labels.
    Args:
        labels_df: All the labels corresponding to the patient ids.
        patient_ids: All the patient ids.
        patient_indices: The indices of the patient ids to include in the dataframe.  This argument is
            used to specify which subset of the patient ids to include in the dataframe
        patient_id_folders: List of folders containing subfolders of patches where each folder corresponds
            to the Whole Slide Image of a patient id.
        subset: train / validation / test
        patches_per_patient_id: Number of patches to samples per patient_id. Defaults to None.
    Returns:
        The newly created dataframe
    """
    df = pd.DataFrame(columns=df_utils.NonBinnedCsvCols.cols)
    for patient_index in patient_indices:
        labels_row = labels_df[labels_df[df_utils.LabelCols.patient_id] == patient_ids[patient_index]]
        assert len(labels_row) == 1, 'Each patient id should correspond to 1 row in label dataframe'
        labels_row = labels_row.iloc[0]

        train_rows = []
        patient_id = patient_ids[patient_index]
        patient_id_folder = patched_patient_folders.get_folder(patient_id, folder_to_patient_ids_to_patches)
        patient_id_path = os.path.join(patient_id_folder, patient_id)
        # add all patches in patient dir to dataframe
        for patch_file_name in folder_to_patient_ids_to_patches[patient_id_folder][patient_id]:
            # values in dataframe row
            image_path = os.path.join(patient_id_path, patch_file_name)
            status_label = labels_row[df_utils.LabelCols.survival_event]
            duration_label = labels_row[df_utils.LabelCols.survival_duration]
            metadata = patient_id

            values = [image_path, status_label, duration_label, subset, metadata]
            train_row = dict(zip(df_utils.NonBinnedCsvCols.cols, values))
            train_rows.append(train_row)
        patient_id_df = pd.DataFrame(train_rows)

        # sample patches
        if patches_per_patient_id is not None and patches_per_patient_id < len(patient_id_df):
            if not do_random_sampling:
                patient_id_df = sample_patches_incrementally(patient_id_df, patches_per_patient_id)
            else:
                patient_id_df = patient_id_df.sample(n=patches_per_patient_id, ignore_index=True)

        df = pd.concat((df, patient_id_df), ignore_index=True)
    return df


def create_dfs(labels_df: pd.DataFrame,
               folder_to_patient_ids_to_patches: Dict[str, Dict[str, str]],
               patched_patient_ids: Set[str],
               split_indices: List[List[List[int]]],
               patches_per_patient_id: Optional[int] = None,
               upsample_valid: bool = False,
               do_random_sampling: bool = False,
               print_summary: bool = False) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """ Creates the train, validation, and test dataframes for each fold of the dataset.
    Args:
        labels_df: All the labels corresponding to the patient ids.
        patient_ids: All the patient ids.
        split_indices:  The entries of the outer list correspond to the split indices for each
            fold.  The entries of each middle list correspond to the patient id indices for each split
            (training / validation / test) of each fold.
        patient_id_folders: List of folders containing subfolders of patches where each folder corresponds
            to the Whole Slide Image of a patient id.
        patches_per_patient_id: Number of patches to samples per patient_id. Defaults to None.
        print_summary: Whether to print a summary or not. Defaults to False.
    Returns:
        Returns training, validation and test dataframes for each fold of the dataset.
    """
    # patient ids that have an associated folder of image patches to the dataframe
    dfs = []
    for i, (train_indices, test_indices) in enumerate(split_indices):
        # create train, valid and test dfs
        train_df = create_df(labels_df,
                             folder_to_patient_ids_to_patches,
                             patched_patient_ids,
                             train_indices,
                             df_utils.SUBSET_TRAIN,
                             patches_per_patient_id,
                             do_random_sampling=do_random_sampling)
        test_df = create_df(labels_df,
                            folder_to_patient_ids_to_patches,
                            patched_patient_ids,
                            test_indices,
                            df_utils.SUBSET_TEST,
                            patches_per_patient_id,
                            do_random_sampling=do_random_sampling)
        # upsample train df
        train_df, valid_df = train_test_split(train_df, test_size=0.15)
        valid_df['subset'] = df_utils.SUBSET_VALID
        upsampled_train_df = df_utils.upsample_df(train_df)
        upsampled_valid_df = df_utils.upsample_df(valid_df)
        upsampled_dfs = [(train_df, upsampled_train_df, 'train'), (valid_df, upsampled_valid_df, 'valid')]
        validate.validate_dfs(upsampled_train_df,
                              upsampled_valid_df,
                              test_df,
                              labels_df,
                              len(patched_patient_ids),
                              i,
                              patches_per_patient_id,
                              print_summary=print_summary)
        validate.validate_upsampled_dfs(upsampled_dfs, print_summary=print_summary)

        # bin outputs
        upsampled_train_df, upsampled_valid_df, test_df = df_utils.bin_outputs(train_df, upsampled_train_df,
                                                                               upsampled_valid_df, test_df)
        validate.validate_binned_outputs(upsampled_train_df, upsampled_valid_df, test_df, print_summary=print_summary)

        # remove extra columns
        upsampled_train_df, upsampled_valid_df, test_df = df_utils.remove_extra_columns(
            (upsampled_train_df, upsampled_valid_df, test_df))
        validate.validate_columns(upsampled_train_df, upsampled_valid_df, test_df)

        dfs.append((upsampled_train_df, upsampled_valid_df, test_df))
    return dfs


def main(args: argparse.ArgumentParser):
    """ Main function """
    # validate arguments and create the output folder in which the csv files will be saved
    validate.validate_args(args)
    utils.create_output_folder(args.output_folder, args.rm_existing_output_dir)

    # get all labels
    labels_df = labels.get_labels(args.labels_folder)
    validate.validate_labels(labels_df, print_summary=args.print_summary)

    # get all patient ids
    folder_to_patient_ids_to_patches, patched_patient_ids = patched_patient_folders.get_folder_to_patient_ids_map(
        args.patient_id_folders, labels_df)
    validate.validate_patches(folder_to_patient_ids_to_patches,
                              patched_patient_ids,
                              labels_df,
                              print_summary=args.print_summary)

    # get split indices for each fold
    split_indices = df_utils.get_split_indices(len(patched_patient_ids), args.num_folds, args.valid_split,
                                               args.test_split)
    validate.validate_split_indices(len(patched_patient_ids), split_indices, print_summary=args.print_summary)

    # create dataframes based on the split indices
    dfs = create_dfs(labels_df,
                     folder_to_patient_ids_to_patches,
                     patched_patient_ids,
                     split_indices,
                     args.patches_per_patient_id,
                     upsample_valid=args.upsample_valid,
                     do_random_sampling=args.do_random_sampling,
                     print_summary=args.print_summary)

    for fold_index, (train_df, valid_df, test_df) in enumerate(dfs):
        file_name = f'fold_{fold_index}.csv'
        csv_path = os.path.join(args.output_folder, file_name)
        df = pd.concat((train_df, valid_df, test_df), ignore_index=True)
        df.to_csv(csv_path)


if __name__ == "__main__":
    utils.set_seeds(SEED)
    parser = get_parser()
    args = parser.parse_args()
    main(args)