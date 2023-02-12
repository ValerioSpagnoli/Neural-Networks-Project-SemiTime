
# This project is based on the following paper
# @inproceedings{fan2021semitime,
#   title        = {Semi-supervised Time Series Classification by Temporal Relation Prediction},
#   author       = {Haoyi Fan, Fengbin Zhang, Ruidong Wang, Xunhua Huang, and Zuoyong Li},
#   booktitle    = {46th International Conference on Acoustics, Speech, and Signal Processing},
#   year         = {2021},
#   organization={IEEE}
# }

# local files
from settings import globalSettings
from data import dataframe, dataset
from models import backboneEncoder, classificationHead, relationHead
from train import train
from test import test

# libraries
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
import argparse

# ------------------------------------------------------------------------------------------------------- #

# Provide the parser for lunch this main from terminal 
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default='CricketX',
                        choices=['CricketX',
                                 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound',
                                 'MFPT', 
                                 'XJTU',
                                 'EpilepticSeizure',],
                        help='dataset')
    
    parser.add_argument('--task', type=str, default='semi-supervised',
                        choices=['supervised', 'semi-supervised'], help='choose method')
    
    parser.add_argument('--run', type=str, default='train',
                        choices=['train', 'test'], help='training or test of model')
    
    parser.add_argument('--save', type=str, default='true',
                        choices=['true', 'false'], help='save the model or not')
    
    opt = parser.parse_args()
    return opt


# Provide the train set of <dataset name>, both in labelled version and unlabelled version and the test set.
def get_dataset(dataset_name=None, split_ratio=None):

    df_train, df_test, num_classes = dataframe.load_dataframe(dataset=dataset_name)
    
    labelledDataset = dataset.LabelledDataset(dataframe=df_train, augmentation=True)
    unlabelledDataset = dataset.UnlabelledDataset(dataframe=df_train, augmentation=True, split_ratio=split_ratio)
    testDataset = dataset.LabelledDataset(dataframe=df_test)

    return labelledDataset, unlabelledDataset, testDataset, num_classes


# SUPERVISED TRAINING
# Model: Backbone Encoder + Classification Head
# Loss: Cross Entropy Loss
# Optimizer: Adam

def supervised(labelledDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None, save=None):

    backbone = backboneEncoder.BackboneEncoder(num_features=num_features).to(device) 
    clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes).to(device) 

    crossEntropy = CrossEntropyLoss().to(device) 
    optimizer = Adam([{'params':backbone.parameters(), 'params':clf_head.parameters()}], lr=learning_rate)

    backbone, clf_head = train.supervised_training(dataset=labelledDataset, backboneEncoder=backbone, classificationHead=clf_head, 
                                       crossEntropy=crossEntropy, optimizer=optimizer,  device=device, setter=setter, save=save)
    
    return backbone, clf_head
        

# SEMI-SUPERVISED TRAINING (SemiTime)
# Model for classification: Backbone Encoder + Classification Head
# Model for relation: Backbone Encoder + Relation Head
# Loss for classification: Cross Entropy Loss
# Loss for relation: Binary Cross Entropy
# Optimizer: Adam (two different Adam optimizer, one for classification and the other for relation)

def semi_supervised(labelledDataset=None, unlabelledDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None, save=None):

    backbone = backboneEncoder.BackboneEncoder(num_features=num_features).to(device)
    clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes).to(device)
    rel_head = relationHead.RelationHead(num_features=num_features).to(device)

    crossEntropy = CrossEntropyLoss().to(device)
    binaryCrossEntropy = BCEWithLogitsLoss().to(device)

    optimizer_clf = Adam([{'params':backbone.parameters(), 'params':clf_head.parameters()}], lr=learning_rate)
    optimizer_rel = Adam([{'params':backbone.parameters(), 'params':rel_head.parameters()}], lr=learning_rate)

    backbone, clf_head = train.semi_supervised_training(labelledDataset=labelledDataset, unlabelledDataset=unlabelledDataset, 
                                            backboneEncoder=backbone, classificationHead=clf_head, relationHead=rel_head,
                                            crossEntropy=crossEntropy, binaryCrossEntropy=binaryCrossEntropy,
                                            optimizer_clf=optimizer_clf, optimizer_rel=optimizer_rel, device=device, setter=setter, save=save)
    
    return backbone, clf_head
    
    
# Perform the test of the model 
def test_model(task=None, dataset_name=None, dataset=None, num_classes=None, num_features=None, backbone=None, clf_head=None):

    crossEntropy = CrossEntropyLoss()

    # if task is supervised 
    if task == 'supervised':
        print(f'[TEST]:')
        print(f' - approach: {task}')
        print(f' - dataset:  {dataset_name}')

        # if isn't passed a backbone, load the backbone saved in checkpoint
        if backbone is None:
            backbone = backboneEncoder.BackboneEncoder(num_features=num_features)
            backbone.load_state_dict(torch.load(f'checkpoints/supervised/{dataset_name}/{dataset_name}_backbone.pt'))

        # if isn't passed a classification head, load the classificaiton head saved in checkpoint
        if clf_head is None:
            clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes)
            clf_head.load_state_dict(torch.load(f'checkpoints/supervised/{dataset_name}/{dataset_name}_classification_head.pt'))

        backbone.to('cpu')
        clf_head.to('cpu')


    elif task == 'semi-supervised':
        print(f'[TEST]:')
        print(f' - approach: {task}')
        print(f' - dataset:  {dataset_name}')

        # if isn't passed a backbone, load the backbone saved in checkpoint
        if backbone is None:
            backbone = backboneEncoder.BackboneEncoder(num_features=num_features)
            backbone.load_state_dict(torch.load(f'checkpoints/semi-supervised/{dataset_name}/{dataset_name}_backbone.pt'))

        # if isn't passed a classification head, load the classificaiton head saved in checkpoint
        if clf_head is None:
            clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes)
            clf_head.load_state_dict(torch.load(f'checkpoints/semi-supervised/{dataset_name}/{dataset_name}_classification_head.pt'))

        backbone.to('cpu')
        clf_head.to('cpu')
        
    else: 
        print('Error: task can be {supervised, semi-supervised}.')
        return
    
    test.test(dataset=dataset, backboneEncoder=backbone, classificationHead=clf_head, crossEntropy=crossEntropy, device='cpu')
    


# ------------------------------------------------------------------------------------------------------- #


def main():

    opt = parse_option()
    dataset_name = opt.dataset
    task = opt.task   
    run = opt.run 
    save_string = opt.save
    save = True
    if save_string == 'False': save=False

    # change this options if necessary
    setter = globalSettings.GlobalSettings(dataset=dataset_name, 
                                           num_folds=8, 
                                           num_epochs=1000,
                                           batch_size=128, 
                                           num_features=64, 
                                           learning_rate=0.01, 
                                           patience=200,
                                           device='mps') # choose here the device ('cpu', 'cuda:0', 'mps')


    dataset_name = setter.__get_settings__(variable='dataset')
    num_features = setter.__get_settings__(variable='num_features')
    learning_rate = setter.__get_settings__(variable='learning_rate')
    split_ratio = setter.__get_settings__(variable='split_ratio')
    device = setter.__get_settings__(variable='device')

    labelledDataset, unlabelledDataset, testDataset, num_classes = get_dataset(dataset_name=dataset_name, split_ratio=split_ratio)

    if run == 'train':

        if task == 'supervised':
            backbone, clf_head = supervised(labelledDataset=labelledDataset, num_classes=num_classes, num_features=num_features, 
                                            learning_rate=learning_rate, device=device, setter=setter, save=save)
        
        elif task == 'semi-supervised':
            backbone, clf_head = semi_supervised(labelledDataset=labelledDataset, unlabelledDataset=unlabelledDataset, num_classes=num_classes, 
                                                 num_features=num_features, learning_rate=learning_rate, device=device, setter=setter, save=save)
    
    elif run == 'test':
        test_model(task=task, dataset_name=dataset_name, dataset=testDataset, num_classes=num_classes, num_features=num_features)



if __name__ == "__main__":
    main()