# RSNA Intracranial Hemorrhage Detection

## Overview

Identify acute intracranial hemorrhage and its subtypes

`https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection`

## Installing useful libraries

```
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```
pip install efficientnet_pytorch
pip install pretrainedmodels
```

## Command

**main.py:**

```
python main.py


 --train True (default:False)

 --post_train True (default:False)
 
 --test True (default:False)

 --foldn 0 (default:0, 0,1,2,3,4)

 --tta 5 (default:5, integer above 0)

 --make_fold True (default:False, 5 folds with seed=10)
```

**windowing.py**

```
python windowing.py

 --mode train (default='test', 'train' or 'test')
 --input /input/csv_dicom/ (directory in which there are csv and dicom)
 --output /../../ (to which save png files)
 --p 4 (default:4,integer depending on your cpu)

```

## Run

1. install libraries in `# Libraries`

2. modify `# Parameters`

3. modify `# Input`

4. train : run command `python main.py --train True --foldn 0`

5. test : run command `python main.py --test True --foldn 0 --tta 5`

6. make folds : run command `python main.py --make_fold True`

## Output

1. output directory : `./output/model_name/`

2. k-fold cross validation : `./fold/`

3. submission csv file : `./submission/`



## Ensemble

**stage2**

1. se_resnet50

Post Train

Test : Fold 0 1 2 3 4

Test time : 22 min / fold (if tta , * 5 )

2. se_resnext50_32x4d

(Post Train)

Test : Fold 0 1 2 3 4

(Stage 2 Dataset Fold3)

Test Time : 44 min / fold (if tta , * 5 )

3. efficientnetb0

Post Train

Test : Fold 1

4. efficientnetb2

Post Train : 43 min

Test : Fold 2


5. se_resnext50_32x4d (512 JPG)

* 6. se_resnext50_32x4d (Stage 2 Dataset)



## Timeline

October 28, 2019 - Entry deadline. You must accept the competition rules before this date in order to compete.

October 28, 2019 - Team Merger deadline. This is the last day participants may join or merge teams.

November 4, 2019 - Stage 1 ends & Model upload deadline.

November 5, 2019 - Stage 2 begins. New test set uploaded.

November 13, 2019 - Stage 2 ends & Final submission deadline.

November 25, 2019 - Solutions & Other Winners Obligations due from winners.

December 1-6, 2019 - RSNA 2019 Conference in Chicago, IL