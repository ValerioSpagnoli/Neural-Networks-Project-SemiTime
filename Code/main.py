
# This project is based on the following paper
# @inproceedings{fan2021semitime,
#   title        = {Semi-supervised Time Series Classification by Temporal Relation Prediction},
#   author       = {Haoyi Fan, Fengbin Zhang, Ruidong Wang, Xunhua Huang, and Zuoyong Li},
#   booktitle    = {46th International Conference on Acoustics, Speech, and Signal Processing},
#   year         = {2021},
#   organization={IEEE}
# }


from settings import globalSettings
from data import dataframe, dataset
from models import backboneEncoder, classificationHead, relationHead
from train import train
from test import test

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
import argparse

# ------------------------------------------------------------------------------------------------------- #

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



def get_dataset(dataset_name=None, split_ratio=None):
    df_train, df_test, num_classes = dataframe.load_dataframe(dataset=dataset_name)
    
    labelledDataset = dataset.LabelledDataset(dataframe=df_train, augmentation=True)
    unlabelledDataset = dataset.UnlabelledDataset(dataframe=df_train, augmentation=True, split_ratio=split_ratio)
    testDataset = dataset.LabelledDataset(dataframe=df_test)

    return labelledDataset, unlabelledDataset, testDataset, num_classes



def supervised(labelledDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None, save=None):
    backbone = backboneEncoder.BackboneEncoder(num_features=num_features).to(device)
    clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes).to(device)

    crossEntropy = CrossEntropyLoss().to(device)
    optimizer = Adam([{'params':backbone.parameters(), 'params':clf_head.parameters()}], lr=learning_rate)

    scores = train.supervised_training(dataset=labelledDataset, backboneEncoder=backbone, classificationHead=clf_head, 
                                       crossEntropy=crossEntropy, optimizer=optimizer,  device=device, setter=setter, save=save)
        


def semi_supervised(labelledDataset=None, unlabelledDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None, save=None):
    backbone = backboneEncoder.BackboneEncoder(num_features=num_features).to(device)
    clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes).to(device)
    rel_head = relationHead.RelationHead(num_features=num_features).to(device)

    crossEntropy = CrossEntropyLoss().to(device)
    binaryCrossEntropy = BCEWithLogitsLoss().to(device)

    optimizer_clf = Adam([{'params':backbone.parameters(), 'params':clf_head.parameters()}], lr=learning_rate)
    optimizer_rel = Adam([{'params':backbone.parameters(), 'params':rel_head.parameters()}], lr=learning_rate)

    scores = train.semi_supervised_training(labelledDataset=labelledDataset, unlabelledDataset=unlabelledDataset, 
                                            backboneEncoder=backbone, classificationHead=clf_head, relationHead=rel_head,
                                            crossEntropy=crossEntropy, binaryCrossEntropy=binaryCrossEntropy,
                                            optimizer_clf=optimizer_clf, optimizer_rel=optimizer_rel, device=device, setter=setter, save=save)
    
    

def test_model(task=None, dataset_name=None, dataset=None, num_classes=None, num_features=None):

    crossEntropy = CrossEntropyLoss()
    classes = {'CricketX': 12, 'UWaveGestureLibraryAll': 8, 'InsectWingbeatSound': 11, 'MFPT': 15, 'XJTU': 15, 'EpilepticSeizure': 5}

    if task == 'supervised':
        print(f'Test of supervised model on dataset {dataset_name}')
        backbone_s = backboneEncoder.BackboneEncoder(num_features=num_features)
        backbone_s.load_state_dict(torch.load(f'checkpoints/supervised/{dataset_name}/{dataset_name}_backbone.pt'))

        clf_head_s = classificationHead.ClassificationHead(num_features=num_features, num_classes=classes[f'{dataset_name}'])
        clf_head_s.load_state_dict(torch.load(f'checkpoints/supervised/{dataset_name}/{dataset_name}_classification_head.pt'))

        test.test(dataset=dataset, num_classes=classes[f'{dataset_name}'], 
                backboneEncoder=backbone_s, classificationHead=clf_head_s, crossEntropy=crossEntropy, scores=None, device='cpu')

    elif task == 'semi-supervised':
        print(f'Test of semi-supervised model on dataset {dataset_name}')
        backbone_ss = backboneEncoder.BackboneEncoder(num_features=num_features)
        backbone_ss.load_state_dict(torch.load(f'checkpoints/semi-supervised/{dataset_name}/{dataset_name}_backbone.pt'))

        clf_head_ss = classificationHead.ClassificationHead(num_features=num_features, num_classes=classes[f'{dataset_name}'])
        clf_head_ss.load_state_dict(torch.load(f'checkpoints/semi-supervised/{dataset_name}/{dataset_name}_classification_head.pt'))

        test.test(dataset=dataset, num_classes=classes[f'{dataset_name}'],
                backboneEncoder=backbone_ss, classificationHead=clf_head_ss, crossEntropy=crossEntropy, scores=None, device='cpu')
    


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
            supervised(labelledDataset=labelledDataset, num_classes=num_classes, num_features=num_features, 
                       learning_rate=learning_rate, device=device, setter=setter, save=save)
        
        elif task == 'semi-supervised':
            semi_supervised(labelledDataset=labelledDataset, unlabelledDataset=unlabelledDataset, num_classes=num_classes, 
                            num_features=num_features, learning_rate=learning_rate, device=device, setter=setter, save=save)
    
    elif run == 'test':
        test_model(task=task, dataset_name=dataset_name, dataset=testDataset, num_classes=num_classes, num_features=num_features)



if __name__ == "__main__":
    main()