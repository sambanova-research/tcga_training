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


from torchvision import models as torchvision_models
from typing import Optional
import itertools
import numpy as np
import torch


class TumorDetector:
    PATCH_SIZE = 512
    THRESHOLD = 0.75

    def __init__(self):
        state = self._get_tumor_detector_state()
        self.tumor_detector = torchvision_models.resnet18()
        self.tumor_detector.fc = torch.nn.Linear(self.tumor_detector.fc.in_features, 2)
        self.tumor_detector.load_state_dict(state['state_dict'])
        self.tumor_detector.eval()

    def _get_tumor_detector_state(self):
        # Please add your tumor detector state here
        return {}

    def _get_tumor_prediction(self, patch: np.ndarray) -> bool:
        """Predict whether or not the specified patch contains a tumor or not.
        Args:
            patch: A smaller square region of a tile.
        Returns:
            True if the patch contains a tumor, False otherwise.
        """
        # Reshape to the shape expected by the tumor detection model
        assert self.tumor_detector is not None, 'Tumor Detector must be initialized'
        torch_patch = torch.Tensor(np.copy(patch)).permute(2, 0, 1).unsqueeze(0)
        err_msg = f'Invalid patch shape passed to tumor detector: {torch_patch.shape}'
        assert tuple(torch_patch.shape) == (1, 3, 512, 512), err_msg
        with torch.no_grad():
            output = self.tumor_detector(torch_patch)
        prediction = bool(torch.argmax(output, dim=-1).item())
        return prediction

    def filter_non_tumor(self, tile: np.ndarray) -> Optional[np.ndarray]:
        """Filter out a tile if the tumor detector predicts the tile to be a non-tumor.
        Args:
            tile:  A square region of the Whole Slide Image.
        Returns:
            The tile if the tile is not filtered out, otherwise None.
        """
        assert tile.shape[0] == tile.shape[1], 'Tile must be a square.'
        assert len(tile.shape) == 3, 'Tile must be 3 dimensional (height, width, channels).'
        for tile_size in tile.shape[:-1]:
            err_msg = f'Tile size {tile_size} must be divisible by patch size {TumorDetector.PATCH_SIZE}'
            assert tile_size % TumorDetector.PATCH_SIZE == 0, err_msg

        # split tile into patches of PATCH_SIZE x PATCH_SIZE, where PATCH_SIZE is the size of patches expected
        # by the tumor detector
        split_size = tile.shape[0] / TumorDetector.PATCH_SIZE
        patches = [np.split(split_tile, split_size, axis=1) for split_tile in np.split(tile, split_size)]
        # flatten list of lists into a single list
        patches = list(itertools.chain.from_iterable(patches))
        assert len(patches) == split_size * split_size, 'Impossible!!! There is some bug in the algorithm.'

        majority_vote = sum(map(self._get_tumor_prediction, patches))
        is_tumor = majority_vote > len(patches) * TumorDetector.THRESHOLD
        return tile if is_tumor else None
