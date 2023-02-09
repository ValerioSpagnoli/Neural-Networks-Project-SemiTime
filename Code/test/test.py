from torch.utils.data import DataLoader
import torch
from test.metrics import accuracy, confusionMatrix


def test(dataset=None, num_classes=None, backboneEncoder=None, classificationHead=None, crossEntropy=None, scores=None, device=None):
    
    test_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    
    backboneEncoder.eval()
    classificationHead.eval()

    with torch.no_grad():

        it = iter(test_loader)
        data = next(it)
        features, targets = data
        features, targets = features.to(device), targets.to(device)

        outputs = backboneEncoder(features)
        outputs = classificationHead(outputs)  

        loss = crossEntropy(outputs, targets)
        loss = loss.item()

    targets = targets.to('cpu')
    outputs = outputs.to('cpu')

    preds_softmax = torch.log_softmax(outputs, dim=1)
    _, preds = torch.max(preds_softmax, dim=1)

    acc = accuracy(predictions=outputs, targets=targets, task='multiclass')
    
    print('Loss: {:.3f}'.format(loss))
    print('Accuracy: {:.3f}%\n'.format(acc*100))

    confusionMatrix(predictions=preds, targets=targets, num_classes=num_classes)