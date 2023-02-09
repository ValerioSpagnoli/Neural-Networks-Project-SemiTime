from settings import globalSettings
from data import dataframe, dataset
from models import backboneEncoder, classificationHead, relationHead
from train import train
from test import test

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
import argparse

# ------------------------------------------------------------------------------------------------------- #

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        choices=['CricketX',
                                 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound',
                                 'MFPT', 
                                 'XJTU',
                                 'EpilepticSeizure',],
                        help='dataset')
    
    parser.add_argument('--model_name', type=str, default='semi-supervised',
                        choices=['supervised', 'semi-supervised'], help='choose method')

    opt = parser.parse_args()
    return opt



def get_dataset(dataset_name=None, split_ratio=None):
    df_train, df_test, num_classes = dataframe.load_dataframe(dataset=dataset_name)
    
    labelledDataset = dataset.LabelledDataset(dataframe=df_train, augmentation=True)
    unlabelledDataset = dataset.UnlabelledDataset(dataframe=df_train, augmentation=True, split_ratio=split_ratio)
    testDataset = dataset.LabelledDataset(dataframe=df_test)

    return labelledDataset, unlabelledDataset, testDataset, num_classes



def supervised(labelledDataset=None, testDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None):
    backbone = backboneEncoder.BackboneEncoder(num_features=num_features).to(device)
    clf_head = classificationHead.ClassificationHead(num_features=num_features, num_classes=num_classes).to(device)

    crossEntropy = CrossEntropyLoss().to(device)
    optimizer = Adam([{'params':backbone.parameters(), 'params':clf_head.parameters()}], lr=learning_rate)

    scores = train.supervised_training(dataset=labelledDataset, backboneEncoder=backbone, classificationHead=clf_head, 
                                       crossEntropy=crossEntropy, optimizer=optimizer,  device=device, setter=setter)
    
    test.test(dataset=testDataset, num_classes=num_classes, 
              backboneEncoder=backbone, classificationHead=clf_head, crossEntropy=crossEntropy, scores=scores, device=device)
    


def semi_supervised(labelledDataset=None, unlabelledDataset=None, testDataset=None, num_classes=None, num_features=None, learning_rate=None, device=None, setter=None):
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
                                            optimizer_clf=optimizer_clf, optimizer_rel=optimizer_rel, device=device, setter=setter)
    
    test.test(dataset=testDataset, num_classes=num_classes, 
            backboneEncoder=backbone, classificationHead=clf_head, crossEntropy=crossEntropy, scores=scores, device=device)
    


# ------------------------------------------------------------------------------------------------------- #


def main():

    opt = parse_option()
    dataset_name = opt.dataset_name
    model_name = opt.model_name    

    setter = globalSettings.GlobalSettings(dataset=dataset_name, 
                                           num_folds=8, 
                                           num_epochs=1000, 
                                           batch_size=128, 
                                           num_features=64, 
                                           learning_rate=0.01, 
                                           patience=200,
                                           device='mps')


    dataset_name = setter.__get_settings__(variable='dataset')
    num_features = setter.__get_settings__(variable='num_features')
    learning_rate = setter.__get_settings__(variable='learning_rate')
    split_ratio = setter.__get_settings__(variable='split_ratio')
    device = setter.__get_settings__(variable='device')

    labelledDataset, unlabelledDataset, testDataset, num_classes = get_dataset(dataset_name=dataset_name, split_ratio=split_ratio)

    if model_name == 'supervised':
        supervised(labelledDataset=labelledDataset, testDataset=testDataset, 
                   num_classes=num_classes, num_features=num_features, learning_rate=learning_rate, device=device, setter=setter)
    
    elif model_name == 'semi-supervised':
        semi_supervised(labelledDataset=labelledDataset, unlabelledDataset=unlabelledDataset, 
                        testDataset=testDataset, num_classes=num_classes, num_features=num_features, learning_rate=learning_rate, device=device, setter=setter)
    

if __name__ == "__main__":
    main()