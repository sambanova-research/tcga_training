
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
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional

import csv_generator.dataframe_utils as df_cols

TCGA_PATIENT_ID_PREFIX = 'TCGA'


def get_folder(patient_id: str, folder_to_patient_ids_to_patches: Dict[str, Dict[str, str]]) -> Optional[str]:
    """ Returns the folder in which the patient_id is located.

    Args:
        patient_id: The patient id
        folder_to_patient_ids: Mapping from folder to the set of patient ids in that folder

    Returns:
        The folder name.  Returns None if the patient id is not in any of the folders.
    """
    for folder, patient_ids_to_patches in folder_to_patient_ids_to_patches.items():
        if patient_id in patient_ids_to_patches.keys():
            return folder
    return None


def get_folder_to_patient_ids_map(patient_id_folders: List[str], labels_df) -> Dict[str, List[str]]:
    """ Creates a dictionary where the key is the patient id folder and the value is
    the list of patient ids in that folder

    Args:
        patient_id_folders: List of folders containing subfolders of patches where each folder corresponds
        to the Whole Slide Image of a patient id

    Returns:
        The mapping from folder to the patient ids in the folder.
    """
    # get all patient ids whose WSI have been processed into image patches
    folder_to_patient_ids_to_patches = {}
    for folder in patient_id_folders:
        patient_id_to_patches = {}
        # get all valid patient ids in the folder
        for patient_id in os.listdir(folder):
            if not patient_id.startswith(TCGA_PATIENT_ID_PREFIX):
                continue
            # if the patient id does not have a corresponding label, skip it
            if patient_id not in set(labels_df[df_cols.LabelCols.patient_id]):
                continue
            patient_id_path = os.path.join(folder, patient_id)
            slide_image_to_patches = defaultdict(list)
            for patch in os.listdir(patient_id_path):
                slide_image = patch.split('_')[0]
                normalcy_id = slide_image.split('-')[3]
                final_id = slide_image.split('-')[5]
                err_msg = f'Invalid final id {final_id} or normalcy id {normalcy_id} found'
                assert len(final_id) == 3 and len(normalcy_id) == 3, err_msg
                # invalid patch
                if 'DX' in final_id or '11' in normalcy_id:
                    continue
                slide_image_to_patches[slide_image].append(patch)

            # find a valid slide image to add the patches to
            patches_lst = list(slide_image_to_patches.values())
            if len(patches_lst) > 0:
                patient_id_to_patches[patient_id] = patches_lst[0]
        folder_to_patient_ids_to_patches[folder] = patient_id_to_patches

    patched_patient_ids = [
        patient_ids_to_patches.keys() for patient_ids_to_patches in folder_to_patient_ids_to_patches.values()
    ]
    patched_patient_ids = list(chain.from_iterable(patched_patient_ids))
    assert len(set(patched_patient_ids)) == len(patched_patient_ids), 'Duplicate patient id folders found'

    return folder_to_patient_ids_to_patches, patched_patient_ids
