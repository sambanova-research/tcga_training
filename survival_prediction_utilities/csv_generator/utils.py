
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
import random
import shutil

import numpy as np
import torch


def create_output_folder(output_folder: str, rm_existing_output_dir: bool):
    """ Creates an output folder, deleting the existing folder if necessary

    Args:
        output_folder: The path to the output folder
        rm_existing_output_dir: Delete the existing output folder if one already exists
    """
    if os.path.exists(output_folder):
        if rm_existing_output_dir:
            shutil.rmtree(output_folder)
        else:
            print('Output folder already exists!  Exiting...')
            exit()
    os.mkdir(output_folder)


def set_seeds(seed: int):
    """ To ensure reproducibility, set a seed for all libraries with any randomness

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
