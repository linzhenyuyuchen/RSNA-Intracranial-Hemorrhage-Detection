#!/usr/bin/env python
# coding: utf-8

# GPU

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Libraries

import datetime
from apex import amp
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate, GaussianBlur, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
import collections
import random
import pickle
from pprint import pprint
import argparse

# Input

dir_csv = '/home/zylin/dataset'
dir_train_img = '/home/zylin/dataset/stage_1_train_images'
dir_test_img = '/home/zylin/dataset/stage_1_test_images'
folds_pkl = os.path.join(dir_csv, 'folds.pkl')

# Parameters

n_classes = 6
n_epochs = 3
batch_size = 28
num_workers = 4
seed = 10
n_fold = 5

# Functions

# parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--make_fold', type=bool, default=False)
    parser.add_argument('--foldn', type=int, default=0)
    parser.add_argument('--tta', type=int, default=5)
    return parser.parse_args()

# Window Functions
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
    return image

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

class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):

        dicom_path = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.dcm')
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        window_center , window_width, intercept, slope = get_windowing(dicom)
        img = rescale_image(image, slope, intercept)
        img = apply_window_policy(img)

        if self.transform:       
            
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            
            return {'image': img}

# k-flod CV

def make_folds(n_fold,seed):

    train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))

    # Split train out into row per image and save a sample

    train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
    train = train[['Image', 'Diagnosis', 'Label']]
    train.drop_duplicates(inplace=True)
    train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
    train['Image'] = 'ID_' + train['Image']

    diagnosis =  ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

    counter_all_label = collections.defaultdict(int)

    for row in train.itertuples():
        for d in diagnosis:
            if getattr(row,str(d)) == 1:
                counter_all_label[d] += 1

    random.seed(seed)
    counter_folds = collections.Counter()
    folds = {}

    for row in train.itertuples():
        labels = [label for label in diagnosis ]#if getattr(row,label)==1
        count_labels = [counter_all_label[label] for label in labels]
        np.array(count_labels)
        min_count_label = labels[np.argmin(count_labels)]
        count_folds = [(f, counter_folds[(f, min_count_label)]) for f in range(n_fold)]
        min_count = min([count for f,count in count_folds])
        fold = random.choice([f for f,count in count_folds if count == min_count])
        folds[row.Image] = fold
        
        for label in labels:
            counter_folds[(fold,label)] += 1

    #pprint(counter_folds)

    with open(folds_pkl, 'wb') as f:
        pickle.dump(folds,f)

    print('folds saved to %s' % folds_pkl)

    return folds


# Model
def model_ll():

    model = torch.hub.load('moskomule/senet.pytorch','se_resnet50',pretrained=True)
    model.fc = torch.nn.Linear(2048, n_classes)

    return model

def run_nn(mode,model,loader,criterion=None,optimizer=None):
    device = torch.device("cuda:0")
    print('mode:%s begin..'% mode)
    if mode in ['train']:
        model.train()
    elif mode in ['valid']:
        model.eval()
    else:
        raise 

    losses = []
    tr_loss = 0
    tk0 = tqdm(loader, desc="Iteration")

    for step, batch in enumerate(tk0):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        with torch.no_grad():
            losses.append(loss.item())
            tr_loss +=loss.item()

        if mode in ['train']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        #if epoch == 1 and step > 6000:
        '''
        if step > 6000:
            epoch_loss = tr_loss / 6000
            print('Loss: {:.4f}'.format(epoch_loss))
            break
        '''
    epoch_loss = tr_loss / len(loader)
    print('Loss: {:.4f}'.format(epoch_loss))

    result = {
        'Loss': epoch_loss,
    }

    print('mode:%s end..' % mode)
    return result


def run_train_valid(foldn):

    train_path = os.path.join(dir_csv, 'train_fold%d.csv') % (foldn)
    valid_path = os.path.join(dir_csv, 'valid_fold%d.csv') % (foldn)

    if not os.path.exists(train_path) or not os.path.exists(valid_path) :
        # load folds
        with open(folds_pkl, 'rb') as f:
            folds = pickle.load(f)
        #pprint(folds)

        '''
        # print count of every fold
        counter_s = collections.Counter()
        for row in folds.itertuples():
            #pprint(row.fold)
            counter_s[row.fold] += 1
        print(counter_s)
        '''

        train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))

        # Split train out into row per image and save a sample

        train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
        train = train[['Image', 'Diagnosis', 'Label']]
        train.drop_duplicates(inplace=True)
        train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
        train['Image'] = 'ID_' + train['Image']

        dcm = glob.glob(os.path.join(dir_train_img, '*.dcm'))
        dcm = [os.path.basename(dcm)[:-4] for dcm in dcm]
        dcm.remove('ID_6431af929')
        dcm = np.array(dcm)

        # prepare the train and valid data
        train_or_val = train[train['Image'].isin(dcm)]


        tmp = train_or_val['Image'].tolist()
        rest = []
        #foldsss = pd.DataFrame.from_dict(folds, orient='index', columns=['ID', 'f'])
        folds.pop('ID_6431af929')

        for ID,f in folds.items():
            if f ==foldn:
                rest.append(ID)
                tmp.remove(ID)

        tmp = np.array(tmp)
        rest = np.array(rest)

        train = train_or_val[train_or_val['Image'].isin(tmp)]
        valid = train_or_val[train_or_val['Image'].isin(rest)]
        train.to_csv(train_path, index=False)
        valid.to_csv(valid_path, index=False)

    # Data loaders

    transform_train = Compose([RandomResizedCrop(height=512, width=512, scale=(0.7,1.0), p=1.0),
        #CenterCrop(200, 200),
        #Resize(224, 224),
        HorizontalFlip(),
        #VerticalFlip(),
        Rotate(limit=30,border_mode=0,p=0.7),
        RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        #ShiftScaleRotate(),
        #GaussianBlur(),
        ToTensor()
    ])

    transform_val = Compose([RandomResizedCrop(height=512, width=512, scale=(0.7,1.0), p=1.0),
        #CenterCrop(200, 200),
        #Resize(224, 224),
        HorizontalFlip(),
        #VerticalFlip(),
        Rotate(limit=30,border_mode=0,p=0.7),
        RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        #ShiftScaleRotate(),
        #GaussianBlur(),
        ToTensor()
    ])

    train_dataset = IntracranialDataset(
        csv_file=train_path, path=dir_train_img, transform=transform_train, labels=True)

    val_dataset = IntracranialDataset(
        csv_file=valid_path, path=dir_train_img, transform=transform_val, labels=True)


    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model = model_ll()

    device = torch.device("cuda:0")
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    plist = [{'params': model.parameters(), 'lr': 6e-5}]
    optimizer = optim.Adam(plist, lr=6e-5)


    scheduler_lr = lr_scheduler.MultiStepLR(optimizer,milestones=[1,2],gamma=2/3,last_epoch=-1)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Train

    for epoch in range(n_epochs):
        
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        result_train = run_nn('train',model,data_loader_train,criterion=criterion,optimizer=optimizer)
        with torch.no_grad():
            result_valid = run_nn('valid',model,data_loader_val,criterion=criterion,optimizer=optimizer)

        #save model

        save_model_path = os.path.join(dir_csv, 'fold%depoch%d.pt') % (foldn,epoch)

        detail = {
            'fold':foldn,
            'epoch':epoch,
            'train loss':result_train['Loss'],
            'valid loss':result_valid['Loss'],
            #'score':score
        }

        torch.save({
            'model':model.state_dict(),
            'optim':optimizer.state_dict(),
            'detail':detail
            }, save_model_path)

        scheduler_lr.step()


def run_test(foldn,ttan):
    epoch = n_epochs-1
    submission_path = os.path.join(dir_csv, 'se_resnet50_submission_fold%d_epoch%dttan%d.csv') % (foldn,epoch,ttan)

    if os.path.exists(submission_path):
        return submission_path

    test_path = os.path.join(dir_csv, 'test.csv') 


    if not os.path.exists(test_path):

        test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))

        test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
        test['Image'] = 'ID_' + test['Image']
        test = test[['Image', 'Label']]
        test.drop_duplicates(inplace=True)

        test.to_csv(test_path, index=False)

    transform_test= Compose([RandomResizedCrop(height=512, width=512, scale=(0.7,1.0), p=1.0),
        #CenterCrop(200, 200),
        #Resize(224, 224),
        HorizontalFlip(),
        Rotate(limit=30,border_mode=0,p=0.7),
        RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        ToTensor()
    ])


    test_dataset = IntracranialDataset(
        csv_file=test_path, path=dir_test_img, transform=transform_test, labels=False)

    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


    save_model_path = os.path.join(dir_csv, 'fold%depoch%d.pt') % (foldn,epoch)


    state = torch.load(save_model_path)
    model = model_ll()
    model.load_state_dict(state['model'])
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    test_pred = np.zeros((len(test_dataset) * n_classes, 1))

    for i, x_batch in enumerate(tqdm(data_loader_test)):
        
        x_batch = x_batch["image"]
        x_batch = x_batch.to(device, dtype=torch.float)
        
        with torch.no_grad():
            
            pred = model(x_batch)
            
            test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
                pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))

    # submission
    submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
    submission.columns = ['ID', 'Label']
    submission_path = os.path.join(dir_csv, 'se_resnet50_submission_fold%d_epoch%dttan%d.csv') % (foldn,epoch,ttan)
    submission.to_csv(submission_path, index=False)
    #submission.head()
    return submission_path

def test_time_augmentation(foldn,tta):
    print('mode test start ..')
    starttime = datetime.datetime.now()
    submission_paths = [run_test(foldn,ttan) for ttan in range(tta)]
    print('mode test end ..')
    endtime = datetime.datetime.now()
    print ('time exhausted %d' % (endtime - starttime).seconds)

    print('mode avg-submission start ..')
    to_path = submission_paths[0].replace("ttan0","tta")
    total = pd.Series(np.zeros(471270))
    for p in submission_paths:
        sub = pd.read_csv(p)
        ids = sub['ID']
        total += sub['Label']
    total /= tta
    submission = pd.DataFrame({'ID':ids,'Label':total})
    submission.to_csv(to_path,index=False)
    print('mode avg-submission end ..')

def main():

    args = get_args()

    if args.train:
        run_train_valid(args.foldn)

    if args.test:
        test_time_augmentation(args.foldn, args.tta)

    if args.make_fold:
        make_folds(n_fold,seed)

if __name__ == '__main__':

    try:
        main()
        #make_folds(5,10)
    except KeyboardInterrupt:
        print('Keyboard Interrupted')



# Plot train example
"""
batch = next(iter(data_loader_train))
fig, axs = plt.subplots(1, 5, figsize=(15,5))


for i in np.arange(5):
    
    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)

"""

# In[23]:


# Plot test example
"""
batch = next(iter(data_loader_test))
fig, axs = plt.subplots(1, 5, figsize=(15,5))

for i in np.arange(5):
    
    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)
"""

