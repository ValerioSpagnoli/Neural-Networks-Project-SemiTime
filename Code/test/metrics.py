import torch

def accuracy(predictions=None, targets=None, task=None):

    targets = targets.to('cpu')
    predictions = predictions.to('cpu')
    
    if task == 'multiclass':

        preds_softmax = torch.log_softmax(predictions, dim=1)
        _, preds =  torch.max(preds_softmax, dim=1)
        correct_preds = (preds == targets).float()
        accuracy = correct_preds.sum()/len(correct_preds)

        return accuracy.item()

    elif task == 'binary':

        preds_sigmoid = torch.round(torch.sigmoid(predictions))
        correct_preds = (preds_sigmoid == targets).float()
        accuracy = correct_preds.sum()/len(correct_preds)

        return accuracy.item()
    



