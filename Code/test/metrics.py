import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    

def confusionMatrix(predictions=None, targets=None, num_classes=None):
    classes = range(num_classes)
    targets = targets.to('cpu')
    predictions = predictions.to('cpu')

    cm = confusion_matrix(targets, predictions, normalize='true')
    plt.figure(figsize=(8,8))
    sns.set(font_scale=1)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=classes, xticklabels=classes)
    plt.title('Confusion Matrix', fontdict={'fontsize':14})
    plt.ylabel('Ground Truth', fontdict={'fontsize':11})
    plt.xlabel('Prediction', fontdict={'fontsize':11})
    plt.show()  


