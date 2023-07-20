Run the following command to generate the training CSVs for Survival Prediction:
1. `pip install -r requirements.txt`
2. `git clone https://github.com/hms-dbmi/rcc_pathology.git`
3. `mv rcc_pathology/survival/ .`
4. `python create_csvs.py --labels-folder <labels_folder> --patient-id-folders <folder_1> <folder_2> --output-folder output --print-summary  --upsample --num-output-bins 2 --patches-per-patient-id 200 --test-split 0.2 --valid-split 0.15 --num-folds 5`
