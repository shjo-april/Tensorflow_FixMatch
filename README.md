# FixMatch

## # Motivation

## # Summary of paper

## # Commands
- for training, 
```sh
# You want to train FixMatch using single gpu.
python train.py --use_gpu 0 --number_of_labels 40 
python train.py --use_gpu 1 --number_of_labels 250 
python train.py --use_gpu 2 --number_of_labels 4000 

# You want to train FixMatch using multiple gpus.
python train.py --use_gpu 0,1,2,3 --number_of_labels 40 
python train.py --use_gpu 0,1,2,3 --number_of_labels 250 
python train.py --use_gpu 0,1,2,3 --number_of_labels 4000 
```

- random seed is 0.

| The number of labels | 40 | 250 | 4000 |
|:---|:---:|:---:|:---:|
| Official implementation (with RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| My implementation (with RA) | 92.39 | 95.14 | 95.62 |


## # References
- Official Tensorflow implementation of "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence" (google-research/fixmatch) [[Code]](https://github.com/google-research/fixmatch)
- Unofficial PyTorch implementation of "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence" [[Code]](https://github.com/kekmodel/FixMatch-pytorch)
- FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence [[Paper]](https://arxiv.org/abs/2001.07685)

