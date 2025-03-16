import math


def adjust_learningrate(optimizer, lr): 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LrShedulerOneNet():

    def __init__(self, dataset_size, batch_size, start_lr = 0.01, finetunestart=0.01, decrease_lr_factor=0.9, 
                decrease_lr_batch=200, first_decrease = 25000, finetunestart_epoch = 35000):
        self.lr = self.start_lr = start_lr
        self.finetunestart = finetunestart
        self.decrease_lr_factor = decrease_lr_factor
        self.decrease_lr_batch = decrease_lr_batch

        self.first_decrease = first_decrease
        self.finetunestart_epoch = finetunestart_epoch

        self.batches_per_epoch = dataset_size//batch_size
        print("lr need epochs:", 60000/self.batches_per_epoch )

        self.info = "LrshedulingOneNet(startlr: %f, finetuning: start %f, step_batch: %i, step_factor %f)"%(
                            self.start_lr, self.finetunestart, self.decrease_lr_batch, self.decrease_lr_factor)

    def adjust_learning_rate(self, optimizer, epoch, batch_count):
        """
        applies adjusted learning rate to the optimizer
        :param optimizer: optimizer for minimizing the net loss
        :param epoch: current epoch of training
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        batch = batch_count+ epoch *self.batches_per_epoch
        if batch > self.first_decrease and batch < self.finetunestart_epoch:
            self.lr = self.start_lr*0.1
        elif batch == self.finetunestart_epoch:
            self.lr = self.finetunestart
        elif batch > self.finetunestart_epoch:
            if batch %self.decrease_lr_batch == 0 :
                self.lr *= self.decrease_lr_factor

        adjust_learningrate(optimizer, self.lr)

class LrShedulerOneNet_paper():

    def __init__(self, dataset_size, batch_size):
        self.lr = 0.01
        self.batches_per_epoch = dataset_size//batch_size
        print("lr need epochs:", 45000/self.batches_per_epoch )
        self.info = "LrshedulingOneNet_paper"

    def adjust_learning_rate(self, optimizer, epoch, batch_count):
        """
        applies adjusted learning rate to the optimizer
        :param optimizer: optimizer for minimizing the net loss
        :param epoch: current epoch of training
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        batch = batch_count+ epoch *self.batches_per_epoch
        if batch <= 25000:
            self.lr = 0.01
        if batch >25000:
             if batch %200 == 0 :
                self.lr *= 0.9
        adjust_learningrate(optimizer, self.lr)


class MetaQNNLrSheduler():

    def __init__(self, max_lr, decrease_lr_factor, decrease_lr_epoch, min_lr=1e-5):
        self.lr = max_lr
        self.decrease_lr_factor = decrease_lr_factor
        self.decrease_lr_epoch = decrease_lr_epoch
        self.min_lr = min_lr
        self.info = "MetaQNNLrSheduler(max_lr: %f, decrease_lr_fractor: %f, decrease_lr_epoch: %f, min_learning_rate: %f)"%(
                                    self.lr, self.decrease_lr_factor, self.decrease_lr_epoch, self.min_lr)
        

    def adjust_learning_rate(self, optimizer, epoch, batch_count):
        """
        applies adjusted learning rate to the optimizer
        :param optimizer: optimizer for minimizing the net loss
        :param epoch: current epoch of training
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        if (epoch > 0 and epoch % self.decrease_lr_epoch == 0 and batch_count == 0):
            tmp_lr = self.lr * self.decrease_lr_factor
            self.lr = tmp_lr if tmp_lr > self.min_lr else self.min_lr
        adjust_learningrate(optimizer, self.lr)

class MetaQNNLrSheduler20epochs(MetaQNNLrSheduler):
    '''MEtaQNN setting designed for 20 epochs'''
    def __init__(self):
        super(MetaQNNLrSheduler20epochs, self).__init__(self, max_learning_rate=0.001, decrease_lr_factor=0.2, decrease_lr_epoch=5)

class MetaQNNLrSheduler150epochs(MetaQNNLrSheduler):
    '''MEtaQNN setting adapted for 150 epochs'''
    def __init__(self):
        super(MetaQNNLrSheduler150epochs, self).__init__(self, max_learning_rate=0.001, decrease_lr_factor=0.5, decrease_lr_epoch=20)

class MetaQNNLrSheduler150epochshighstartlr(MetaQNNLrSheduler):
    '''MEtaQNN setting adapted for 150 epochs with start lr =0.01'''
    def __init__(self):
        super(MetaQNNLrSheduler150epochshighstartlr, self).__init__(self, max_learning_rate=0.01, decrease_lr_factor=0.5, decrease_lr_epoch=15)


class LRShedulerfromList():

    def __init__(self, lrlist):
        '''
        learning rate sheduler according to list in form
        [(number of epochs0, lr0),... ,(number of epochx,lrx)]
        '''
        self.lrlist = lrlist
        self.lr = self.lrlist[0][1]
        self.epoch_lrstart = 0 #epoch at which lr has changed
        self.info = "LearningRateSchedulerCODEBRIM([(epochs, lr)]: %s)"%(str(self.lrlist))


    def adjust_learning_rate(self, optimizer, epoch, batch_count):
        """
        applies adjusted learning rate to the optimizer
        :param optimizer: optimizer for minimizing the net loss
        :param epoch: current epoch of training
        :param batch_count: counter for batches in an epoch
        :return adjusted learning rate
        """
        self.lr = self.lrlist[0][1]
        if ((epoch - self.epoch_lrstart)-1 >= self.lrlist[0][0]):
            if len(self.lrlist) > 1:
                self.lrlist.pop(0)
                self.epoch_lrstart = epoch

        adjust_learningrate(optimizer, self.lr)
        
class MetaQNNCIFARSheduling(LRShedulerfromList):
    def __init__(self):
        super(MetaQNNCIFARSheduling, self).__init__(lrlist =[(40, 0.025),(40,0.0125),(160, 0.0001), (60,0.00001)])

class FixedLRSheduling150epochs(LRShedulerfromList):
    def __init__(self):
        super(FixedLRSheduling150epochs, self).__init__(lrlist =[(20, 0.01),(30, 0.001),(40,0.0001),(60,0.00001)])


class FixedLR():
    def __init__(self, lr):
        self.lr = lr
        self.info = F"Fixed learningrate {lr}"

    
    def adjust_learning_rate(self, optimizer, epoch, batch_count):
        #adjust_learningrate(optimizer, self.start_lr) #not neccessary if nothing else changes lr
        pass