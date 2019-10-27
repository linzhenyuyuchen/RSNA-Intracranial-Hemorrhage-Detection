# RSNA Intracranial Hemorrhage Detection

## Installing useful libraries

```
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Command

**se_resnet50.py:**

```
python se_resnet50.py


 --train True (default:False)
 
 --foldn 0 (default:0, 0,1,2,3,4)

 --test True (default:False)

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

2. modify `# Input`

3. modify `# Parameters`

## Timeline
October 28, 2019 - Entry deadline. You must accept the competition rules before this date in order to compete.

October 28, 2019 - Team Merger deadline. This is the last day participants may join or merge teams.

November 4, 2019 - Stage 1 ends & Model upload deadline*.

November 5, 2019 - Stage 2 begins. New test set uploaded.

November 11, 2019 - Stage 2 ends & Final submission deadline.

November 25, 2019 - Solutions & Other Winners Obligations due from winners.

December 1-6, 2019 - RSNA 2019 Conference in Chicago, IL