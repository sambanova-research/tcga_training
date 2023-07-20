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


import staintools


class StainNormalizer:
    METHOD = 'vahadane'

    def __init__(self):
        """Instantiate the stain normalizer."""
        target_image = self.get_stain_normalizer_image()
        target_standardized_image = staintools.LuminosityStandardizer.standardize(target_image)
        self.normalizer = staintools.StainNormalizer(method=StainNormalizer.METHOD)
        self.normalizer.fit(target_standardized_image)

    def get_stain_normalizer_image(self):
        # NOTE: Please add your normalization image here.
        return -1

    def apply_stain_normalizer(self, tile):
        """Apply stain normalization to the tile.

        Args:
            tile:  A square region of the Whole Slide Image.
        Returns:
            The stain normalized tile.
        """
        standardized_tile = staintools.LuminosityStandardizer.standardize(tile)
        # Stain normalize
        try:
            stain_normalized_tile = self.normalizer.transform(standardized_tile)
        except Exception as exc:
            print(f'Exception thrown {exc}')
            return None
        return stain_normalized_tile
