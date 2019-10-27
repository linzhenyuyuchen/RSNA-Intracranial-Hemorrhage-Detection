#!/usr/bin/env python
# coding: utf-8

# Libraries

import os
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import functools
import argparse

# Functions

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--p', type=int, default=4)
    return parser.parse_args()

def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def rescale_image(image, slope, intercept):
    return image * slope + intercept

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

def apply_window_policy(image):
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    #print(image.shape)
    return image

def apply_scale(image):
    imin = image.min()
    imax =  image.max()
    image = (image - imin) / (imax - imin)
    return image

def conv(image_path,output):
    to_path = os.path.join(output,os.path.basename(image_path))
    if not os.path.exists(to_path):
        data = pydicom.dcmread(image_path)
        window_center , window_width, intercept, slope = get_windowing(data)
        img = pydicom.read_file(image_path).pixel_array
        img_do = rescale_image(img, slope, intercept)
        img_do = apply_window_policy(img_do)
        to_path = to_path.replace(".dcm",".png")
        img_do = apply_scale(img_do)
        plt.imsave(to_path, img_do, format="png")

def main():

    args = get_args()
    # /home/user1012/.dataset/
    train_path = args.input + '/stage_1_train_images/*.dcm'
    test_path = args.input + '/stage_1_test_images/*.dcm'
    train = sorted(glob(train_path))
    test = sorted(glob(test_path))

    if args.mode in ['train']:
        raw_input = train
    if args.mode in ['test']:
        raw_input = test
    with Pool(processes=args.p) as pool:
        records = list(tqdm(iterable=pool.imap_unordered(
            functools.partial(conv,output=args.output),
            raw_input.items()
        ),
        total=len(raw_input),
        ))
    return records


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')


