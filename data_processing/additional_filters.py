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


import numpy as np
from typing import Optional, Tuple


MIN_RED_VALUE = 128
MAX_GREEN_VALUE = 220
RED_GREEN_DIFF_VALUE = 10
PINK_THRESHOLD = 0.9


def is_not_pink(color: Tuple[int, int, int]) -> bool:
    """Whether a color is 'pinkish / purplish' or not.
    Most of the tissue regions in the Whole Slide Images are 'pinkish / purplish'.
    Args:
        color: Red, Green, Blue channel values for a color ranging from 0 to 255.
    Returns:
        True if the color is not pinkish / purplish, False otherwise.
    """
    is_red_high = color[0] >= MIN_RED_VALUE
    is_green_low = color[1] <= MAX_GREEN_VALUE
    is_green_much_less_than_red = color[1] < color[0] - RED_GREEN_DIFF_VALUE
    is_green_less_than_blue = color[1] < color[2]
    is_color_pink = is_red_high and is_green_low and is_green_much_less_than_red and is_green_less_than_blue
    return not is_color_pink


def filter_by_pixel_values(tile: np.ndarray) -> Optional[np.ndarray]:
    """Filter out a tile if most of the pixels look not pink/purple.

    Args:
        tile:  A square region of the Whole Slide Image.
    Returns:
        The tile if the tile is not filtered out, otherwise None.
    """
    invalid_pixels = np.apply_along_axis(is_not_pink, 2, tile)
    threshold = PINK_THRESHOLD

    percent_invalid_pixels = np.sum(invalid_pixels) / (tile.shape[0] * tile.shape[1])
    return tile if percent_invalid_pixels <= threshold else None
