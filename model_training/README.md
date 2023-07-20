# TCGA model training 

This repository releases dataset, model training and evaluation code used for producing results for High-resolution Integrated Pathology. 

## Installation
`pip install -r requirements.txt`

## Usage

<b>Training (for single GPU) </b>
```
python -m torch.distributed.launch --nproc_per_node=1 --master_addr=localhost --master_port=12323 --use_env hook.py --data-dir --dataset-csv-path --in-height 512 --in-width 512 --drop-conv 0.0 --drop-fc 0.0 --num-classes 3 --epochs 30 --batch-size 2 --model rescale50 --mode train --num-workers 2 --learning-rate 1e-4 --seed 512 --weight-decay 0.001 --log-dir --use-ddp --optimizer adamw --local_rank 0
```

<b> Offline Validation </b>
```
python hook.py --data-dir --dataset-csv-path --in-height 512 --in-width 512 --drop-conv 0.0 --drop-fc 0.0 --num-classes 3 --batch-size 2 --model rescale50 --mode validation --num-workers 2 --ckpt-dir --inference 
```

<b> Inference </b>
```
python hook.py --data-dir --dataset-csv-path --in-height 512 --in-width 512 --drop-conv 0.0 --drop-fc 0.0 --num-classes 3 --batch-size 2 --model rescale50 --mode predict --num-workers 2 --seed 512 --ckpt-file --log-dir --inference
```

<b> Metric Evaluation </b>

<b> Note: </b> artifact-dir is the log dir obtained from running inference. 
```
python evaluation/metric.py --artifact-dir --log-dir
```
