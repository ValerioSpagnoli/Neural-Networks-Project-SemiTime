from utils.earlyStopping import EarlyStopping
from test.metrics import accuracy
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader


def supervised_training(dataset=None, backboneEncoder=None, classificationHead=None, crossEntropy=None, optimizer=None, device=None, setter=None):

    num_epochs = setter.__get_settings__(variable='num_epochs')
    patience = setter.__get_settings__(variable='patience')
    batch_size = setter.__get_settings__(variable='batch_size')
    num_folds = setter.__get_settings__(variable='num_folds')
    dataset_name = setter.__get_settings__(variable='dataset')

    total_train_loss=[]
    total_train_acc=[]
    total_val_loss=[]
    total_val_acc=[]

    Kfolder = KFold(n_splits = num_folds, shuffle = True)
    earlyStopping = EarlyStopping(patience=patience)
    stop = False

    print('---------------------------------------------------------------------------------------------')
    print(f'START SUPERVISED TRAINING -- number of folds: {num_folds}, number of epochs: {num_epochs}, patience: {patience}')
    print('---------------------------------------------------------------------------------------------\n')

    
    for fold, (train_idx, val_idx) in enumerate(Kfolder.split(np.arange(len(dataset)))): 

        earlyStopping.__restart__()
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_sampler)

        print(f'Fold: [{fold+1}/{num_folds}]')

        for e in range(num_epochs):

            backboneEncoder.train()
            classificationHead.train()

            for i, data in enumerate(train_loader):
                features, targets = data
                features, targets = features.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = backboneEncoder(features)
                outputs = classificationHead(outputs)  

                loss = crossEntropy(outputs, targets)
                loss.backward()
                optimizer.step()

                total_train_loss.append(loss.item())
                total_train_acc.append(accuracy(predictions=outputs, targets=targets, task='multiclass'))

            train_loss = np.mean(total_train_loss)
            train_acc = np.mean(total_train_acc)*100


            backboneEncoder.eval()
            classificationHead.eval()

            with torch.no_grad():

                for i, data in enumerate(val_loader):
                    features, targets = data
                    features, targets = features.to(device), targets.to(device)

                    outputs = backboneEncoder(features)
                    outputs = classificationHead(outputs)  

                    loss = crossEntropy(outputs, targets)

                    total_val_loss.append(loss.item())
                    total_val_acc.append(accuracy(predictions=outputs, targets=targets, task='multiclass'))

            val_loss = np.mean(total_val_loss)
            val_acc = np.mean(total_val_acc)*100

            print(f'[{fold+1}/{num_folds}][{e+1}/{num_epochs}]:')
            print('  - training:      loss = {:.3f}, accuracy = {:.3f}%'.format(train_loss, train_acc))
            print('  - validation:    loss = {:.3f}, accuracy = {:.3f}%'.format(val_loss, val_acc))


            counter, stop, min_loss, min_epoch = earlyStopping.__check__(loss=val_loss, epoch=e+1)
            if not stop:
                print('  - earlyStopping: counter = {:d}, min_loss = {:.3f}, min_epoch = {:d}\n'.format(counter, min_loss, min_epoch))
            else:
                print('  - earlyStopping: counter = {:d}, min_loss = {:.3f}, min_epoch = {:d}'.format(counter, min_loss, min_epoch))
                print('Stop training on fold {:d} at epoch {:d}\n'.format(fold+1, e+1))
                break

            torch.save(backboneEncoder.state_dict(), f'./checkpoints/supervised/{dataset_name}/{dataset_name}_backbone.pt')
            torch.save(classificationHead.state_dict(), f'./checkpoints/supervised/{dataset_name}/{dataset_name}_classification_head.pt')

    torch.save(backboneEncoder.state_dict(), f'./checkpoints/supervised/{dataset_name}/{dataset_name}_backbone.pt')
    torch.save(classificationHead.state_dict(), f'./checkpoints/supervised/{dataset_name}/{dataset_name}_classification_head.pt')

    print('---------------------------------------------------------------------------------------------')
    print(f'FINISH SUPERVISED TRAINING')
    print('Average train loss:      {:.3f} -- Average train accuracy:      {:.3f}%'.format(np.mean(total_train_loss), np.mean(total_train_acc)*100))
    print('Average validation loss: {:.3f} -- Average validation accuracy: {:.3f}%'.format(np.mean(total_val_loss), np.mean(total_val_acc)*100))
    print('---------------------------------------------------------------------------------------------')

    scores = {'train_loss':total_train_loss, 'train_acc':total_train_acc, 'validation_loss':total_val_loss, 'validation_acc':total_val_acc}
    return scores






def semi_supervised_training(labelledDataset=None, unlabelledDataset=None, backboneEncoder=None, classificationHead=None, relationHead=None, crossEntropy=None, binaryCrossEntropy=None, optimizer_clf=None, optimizer_rel=None, device=None, setter=None):

    num_epochs = setter.__get_settings__(variable='num_epochs')
    patience = setter.__get_settings__(variable='patience')
    batch_size = setter.__get_settings__(variable='batch_size')
    num_folds = setter.__get_settings__(variable='num_folds')
    dataset_name = setter.__get_settings__(variable='dataset')

    Kfolder = KFold(n_splits = num_folds, shuffle = True)
    earlyStopping = EarlyStopping(patience=patience)
    stop = False

    total_train_loss_clf = []
    total_train_loss_rel = []
    total_train_acc_clf = []
    total_train_acc_rel = []

    total_val_loss = []
    total_val_acc = []

    print('---------------------------------------------------------------------------------------------')
    print(f'START SEMI-SUPERVISED TRAINING -- number of folds: {num_folds}, number of epochs: {num_epochs}, patience: {patience}')
    print('---------------------------------------------------------------------------------------------\n')

    
    for fold, (train_idx, val_idx) in enumerate(Kfolder.split(np.arange(len(labelledDataset)))): 

        earlyStopping.__restart__()

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader_lab = DataLoader(dataset=labelledDataset, batch_size=batch_size, sampler=train_sampler)
        val_loader_lab = DataLoader(dataset=labelledDataset, batch_size=batch_size, sampler=val_sampler)
        train_loader_unlab = DataLoader(dataset=unlabelledDataset, batch_size=batch_size, shuffle=False)

        print(f'Fold: [{fold+1}/{num_folds}]')

        for e in range(num_epochs):

            #------------------------------#
            # TRAINING 
            #------------------------------#
            
            backboneEncoder.train()
            classificationHead.train()
            relationHead.train()

            epoch_train_loss={'loss_clf':[], 'loss_rel':[]}
            epoch_train_acc={'acc_clf':[], 'acc_rel':[]}

            # Training on labelled dataset
            for i, data_lab in enumerate(train_loader_lab):
                features, targets = data_lab
                features, targets = features.to(device), targets.to(device)

                optimizer_clf.zero_grad()

                outputs = backboneEncoder(features)
                outputs = classificationHead(outputs)  

                loss = crossEntropy(outputs, targets)

                loss.backward()
                optimizer_clf.step()

                epoch_train_loss['loss_clf'].append(loss.item())
                epoch_train_acc['acc_clf'].append(accuracy(predictions=outputs, targets=targets, task='multiclass'))

            
            # Training on unlabelled dataset
            for i, data_unlab in enumerate(train_loader_unlab):
                past_features, future_features = data_unlab
        
                roll_param = int(np.random.randint(len(future_features))*0.75) #roll 
                pos_features = future_features
                neg_features = torch.roll(future_features, roll_param, dims=0)

                past_features = past_features.to(device)
                pos_features = pos_features.to(device)
                neg_features = neg_features.to(device)

                optimizer_rel.zero_grad()

                output_past = backboneEncoder(past_features)
                output_pos = backboneEncoder(pos_features)
                output_neg = backboneEncoder(neg_features)
                
                pos_segments = torch.cat((output_past, output_pos), dim=1)
                neg_segments = torch.cat((output_past, output_neg), dim=1)

                output_pos = relationHead(pos_segments)
                output_neg = relationHead(neg_segments)

                target_pos = torch.ones(output_pos.shape).to(torch.float32).to(device)
                target_neg = torch.zeros(output_neg.shape).to(torch.float32).to(device)

                outputs = torch.cat((output_pos, output_neg), dim=0).squeeze()
                targets = torch.cat((target_pos, target_neg), dim=0).squeeze()

                loss = binaryCrossEntropy(outputs, targets)

                loss.backward()
                optimizer_rel.step()

                epoch_train_loss['loss_rel'].append(loss.item())
                epoch_train_acc['acc_rel'].append(accuracy(predictions=outputs, targets=targets, task='binary'))


            loss_clf = np.mean(epoch_train_loss['loss_clf'])
            loss_rel = np.mean(epoch_train_loss['loss_rel'])
            acc_clf = np.mean(epoch_train_acc['acc_clf'])
            acc_rel = np.mean(epoch_train_acc['acc_rel'])


            #------------------------------#
            # VALIDATION
            #------------------------------#

            backboneEncoder.eval()
            classificationHead.eval()

            epoch_val_loss=[]
            epoch_val_acc=[]

            with torch.no_grad():

                for i, data in enumerate(val_loader_lab):
                    features, targets = data
                    features, targets = features.to(device), targets.to(device)

                    outputs = backboneEncoder(features)
                    outputs = classificationHead(outputs)  

                    loss = crossEntropy(outputs, targets)

                    epoch_val_loss.append(loss.item())
                    epoch_val_acc.append(accuracy(predictions=outputs, targets=targets, task='multiclass'))

            val_loss = np.mean(epoch_val_loss)
            val_acc = np.mean(epoch_val_acc)


            #------------------------------#
            # RESULTS + CHECK EARLY STOPPING
            #------------------------------#


            print(f'[{fold+1}/{num_folds}][{e+1}/{num_epochs}]:')
            print('  - training:      loss_clf = {:.3f}, acc_clf = {:.3f}% -- loss_rel = {:.3f}, acc_rel = {:.3f}%'.format(loss_clf, acc_clf*100, loss_rel, acc_rel*100))
            print('  - validation:    loss = {:.3f}, accuracy = {:.3f}%'.format(val_loss, val_acc*100))

            total_train_loss_clf.append(loss_clf)
            total_train_loss_rel.append(loss_rel)
            total_train_acc_clf.append(acc_clf)
            total_train_acc_rel.append(acc_rel)

            total_val_loss.append(val_loss)
            total_val_acc.append(val_acc)


            counter, stop, min_loss, min_epoch = earlyStopping.__check__(loss=val_loss, epoch=e+1)
            if not stop:
                print('  - earlyStopping: counter = {:d}, min_loss = {:.3f}, min_epoch = {:d}\n'.format(counter, min_loss, min_epoch))
            else:
                print('  - earlyStopping: counter = {:d}, min_loss = {:.3f}, min_epoch = {:d}'.format(counter, min_loss, min_epoch))
                print('Stop training on fold {:d} at epoch {:d}\n'.format(fold+1, e+1))
                break

            torch.save(backboneEncoder.state_dict(), f'./checkpoints/semi-supervised/{dataset_name}/{dataset_name}_backbone.pt')
            torch.save(classificationHead.state_dict(), f'./checkpoints/semi-supervised/{dataset_name}/{dataset_name}_classification_head.pt')
    
    torch.save(backboneEncoder.state_dict(), f'./checkpoints/semi-supervised/{dataset_name}/{dataset_name}_backbone.pt')
    torch.save(classificationHead.state_dict(), f'./checkpoints/semi-supervised/{dataset_name}/{dataset_name}_classification_head.pt')

    print('---------------------------------------------------------------------------------------------')
    print(f'FINISH SEMI-SUPERVISED TRAINING')
    print('Average train loss classifier: {:.3f} -- Average train accuracy classifier: {:.3f}%'.format(np.mean(total_train_loss_clf), np.mean(total_train_acc_clf)*100))
    print('Average train loss relation:   {:.3f} -- Average train accuracy relation:   {:.3f}%'.format(np.mean(total_train_loss_rel), np.mean(total_train_acc_rel)*100))
    print('Average validation loss:       {:.3f} -- Average validation accuracy:       {:.3f}%'.format(np.mean(total_val_loss), np.mean(total_val_acc)*100))
    print('---------------------------------------------------------------------------------------------')


    scores = {'train_loss_clf':total_train_loss_clf, 'train_acc_clf':total_train_acc_clf, 'train_loss_rel':total_train_loss_rel, 
              'train_acc_rel':total_train_acc_rel, 'validation_loss':total_val_loss, 'validation_accuracy':total_val_acc}

    return scores