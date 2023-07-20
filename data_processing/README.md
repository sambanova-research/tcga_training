# Data Processing Guide
It isn't possible to directly train on the high-resolution Whole Slide Image because our hardware will run out of memory. To solve this problem, we: 

1. Split up the Whole Slide Image (WSI) into smaller tiles. Each tile is treated as a distinct example in our dataset and the label assigned to each tile is the same as the label of the original Whole Slide Image the tile came from. 
2. Train our models on the tiles.  

Follow this guide to process your WSI files.

## Downloading TCGA WSIs
If the WSI you're preprocessing are not TCGA WSI, skip this step and go to the Preprocessing step.

1. Download the TCGA client from here: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
2. Download a manifest file: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/
3. Filter the lines in the manifest file, keeping the lines corresponding to the WSIs you want to download.
4. Create a slides_folder directory where your WSIs will be downloaded: `mkdir <slides_folder>`.
5. Download the WSIs: `./gdc-client download -m manifest.txt -d <slides_folder>`

## Preprocess the WSIs
To preprocess the WSIs, we used code from this repository: https://github.com/MarvinLer/tcga_segmentation

1. Set the tile size.  If you want tile size `x`, open the file `code/settings.py` and set the desired overlap to be
`desired_overlap = (x - desired_tile_width) / 2`.
2. Run the following command:
```
python code/data_processing/main.py --no-download --source-slides-folder <slides_folder>
```
3. Check that your tiles are in the output folder.
<!-- Where do I define the output folder? What should I expect to see in the output folder? -->

## Additional Filters
By default the MarvinLer repository linked above supports a limited number of filters such as the `is_tile_mostly_background` function in this file: https://github.com/MarvinLer/tcga_segmentation/blob/master/code/data_processing/svs_factory.py#L111

We added a few more filters as part of our data processing to filter out tiles that were not 'pink' (i.e. the color of the tissue).  We include some of these functions in `additional_filters.py` in our repository.

## Tumor Detection
We additionally filtered tiles that didn't have any tumors.  To do this we passed our tiles through a tumor detection model and only kept the tiles if the tumor detection model predicted that the tile contained a tumor.  Our code is located in `tumor_detector.py`.

## Stain Normalization
Finally, we applied stain normalization to our tiles using this repository: https://github.com/Peter554/StainTools.  Our code is located in `stain_normalization.py`.
