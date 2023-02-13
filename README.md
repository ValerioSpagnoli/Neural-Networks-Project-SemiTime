# Neural Networks Project -- SemiTime

This repository contains the Project of the Neural Networks course of the master in Artificial Intelligence and Robotics at La Sapienza University in Rome.

The project has been based on this paper: **Semi-supervised time series classification by temporal relation prediction** (https://haoyfan.github.io/papers/SemiTime_ICASSP2021.pdf), which describes a novel approach for time series classification.
In this work the authors have proposed a method of semi-supervised time series classification architecture (termed as **SemiTime**) by gaining from the structure of unlabeled data in a self-supervised manner. 

|Schematic illustration of semi-supervised techniques described| SemiTime architecture|
|--------|--------|
|  ![schematic_illustration](./images/schematic_illustration.png)  |  ![SemiTime_architecture](./images/SemiTime_architecture.png)  |


## Requirements 
* Python 3.9.6
* torch 1.13.1
* torchvision 0.14.1
* numpy 1.24.1
* sklearn 1.2.1
* pandas 1.5.3


## Run model

### Training

**Supervised**:

```bash 
cd Code
python main.py --dataset CricketX --task supervised --run train --save true
```

**Semi-Supervised**:
```bash
cd Code
python main.py --dataset CricketX --task semi-supervised --run train --save true
```
After training the model is saved in 'checkpoints' folder.


### Test
The model is automatically tested after training. If you want to test it later: 

**Supervised**:
```bash
cd Code
python main.py --dataset CricketX --task supervised --run test
```

**Semi-Supervised**:
```bash
cd Code
python main.py --dataset CricketX --task semi-supervised --run test
```

or you can also use the provided notebook.

### Parse options:
- dataset: {CricketX, UWaveGestureLibraryAll, InsectWingbeatSound, MFPT, XJTU, EpilepticSeizure} 
- task: {supervised, semi-supervised} 
- run: {train, test}
- save: {true, false} (save or not the model in ./Code/checkpoint/{task}/{dataset})
