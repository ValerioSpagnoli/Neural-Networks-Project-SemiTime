class EarlyStopping():
    def __init__(self, patience=None):
        super().__init__()
        self.patience=patience #number of epochs with loss>=min_loss without stopping
        self.counter=1 #counter of epochs with abs(min_loss-loss) <= threshold
        self.min_loss=0 #loss of previous epoch
        self.min_epoch=1
        self.stop = False
        self.init = True

    def __restart__(self):
        self.counter=1
        self.min_loss=0
        self.min_epoch=1
        self.stop=False
        self.init=True
        return

    def __check__(self, loss=None, epoch=None):

        if self.init:
            self.min_loss=loss
            self.min_epoch=epoch
            self.counter+=1
            self.stop=False
            self.init=False
            return self.counter, self.stop, self.min_loss, self.min_epoch

        elif self.counter==self.patience: 
            self.stop=True
            self.init=True
            return self.counter, self.stop, self.min_loss, self.min_epoch

        else:   
            if loss >= self.min_loss:
                self.counter+=1
                self.stop=False
            
            else:
                self.min_loss=loss
                self.min_epoch=epoch
                self.counter=1
                self.stop=False

            return self.counter, self.stop, self.min_loss, self.min_epoch