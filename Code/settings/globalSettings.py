import torch.backends.mps as mps
import torch.cuda as cuda

class GlobalSettings():

    def __init__(self, dataset=None, num_folds=None, num_epochs=None, batch_size=None, num_features=None, learning_rate=None, patience=None, split_ratio=None, device=None):
        
        self.dataset='CricketX'
        self.num_folds=8
        self.num_epochs=1000
        self.batch_size=128
        self.num_features=64
        self.learning_rate=0.01
        self.patience=200
        self.split_ratio=0.4
        self.device='cpu'

        if dataset is not None:
            self.dataset = dataset

        if num_folds is not None:
            self.num_folds = num_folds
     
        if num_epochs is not None:
            self.num_epochs = num_epochs

        if batch_size is not None:
            self.batch_size = batch_size

        if num_features is not None:
            self.num_features = num_features

        if learning_rate is not None:
            self.learning_rate = learning_rate

        if patience is not None:
            self.patience = patience

        if split_ratio is not None:
            self.split_ratio = split_ratio
        
        if device is not None:
            if device=='cpu': self.device='cpu'
            elif device=='mps':
                if mps.is_available(): self.device='mps' 
                else: self.device='cpu'
            elif device=='cuda': 
                if cuda.is_available(): self.device='cuda:0'
                else: self.device='cpu'

    
    def __get_settings__(self, variable=None):

        if variable == 'dataset':
            return self.dataset
        
        elif variable == 'num_folds':
            return self.num_folds
        
        elif variable == 'num_epochs':
            return self.num_epochs
        
        elif variable == 'batch_size':
            return self.batch_size
        
        elif variable == 'num_features':
            return self.num_features
        
        elif variable == 'learning_rate':
            return self.learning_rate
        
        elif variable == 'patience':
            return self.patience
        
        elif variable == 'split_ratio':
            return self.split_ratio
        
        elif variable == 'device':
            return self.device
        
        else:
           return {'dataset':self.dataset,
                   'num_fold':self.num_folds, 
                   'num_epochs':self.num_epochs, 
                   'batch_size':self.batch_size, 
                   'num_features':self.num_features, 
                   'learning_rate':self.learning_rate,
                   'patience':self.patience,
                   'split_ratio':self.split_ratio,
                   'device':self.device}

