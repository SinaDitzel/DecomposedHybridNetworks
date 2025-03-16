from time import gmtime,strftime, localtime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from device import device

class Logger():
    def __init__(self, name = "", debug = False, path = './runs/'):
        print("Logger init")
        self.path = path +  strftime("%Y-%m-%d_%H-%M-%S", localtime()) +name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.logfile = os.path.join(self.path, 'training'+name+'.log')
        print("save logs to file: ", self.logfile)

        self.writer = SummaryWriter(self.path)
        self.debug = debug

    def add(self, text):
        print("log:", text)
        with open(self.logfile, 'a') as f:
            f.write(str(text)+"\n")
    
    def add_data(self, label, value, i, addTensorboard =True):
        self.writer.add_scalar(label, value, i)   
        if isinstance(value, float):
            if value > 0.01:    
                self.add(label +" : %.3f"%(value))
            else:    
                self.add(label +" : %.5f"%(value))   
        else:
            self.add(label +" : "+str(value))   
    
    def add_datas(self, tag, dict, i):
        self.writer.add_scalars(tag, dict,i)

    def add_txt(self, text, topic ="", i=0):
        self.writer.add_text(topic, text, i)
        self.add(text)

    def save_model(self, model):
        torch.save(model.state_dict(), self.path+'/model_state_dict')

    def load_model(self):
        return torch.load(self.path+'/model_state_dict')

    def log_flat_weights(self, step, weight):
        
        '''a1, a2= a1.numpy(), a2.numpy()
        min_val = min(min(a1), min(a2))
        max_val = max(max(a1), max(a2))
        for a,text in [(a1,text1),(a2,text2)]:
            print(a.shape)
            x_old, y_old, c_old = a.shape
            for x in range(int(np.sqrt(c_old)),c_old):
                if (c_old%x == 0):
                    x_new = x *2
                    y_new = c_old/x *2
            b = np.zeros((x_new, y_new))
            for j in range(0, int(y_new/y_old)):
                for i in range(0,int(x_new/x_old)):
                    c  =  ((i))+((x_new/x_old)*(j))
                    b[i*x_old:i*x_old+x_old,j*y_old:j*y_old+y_old]=a[:,:,int(c)]
            print(b.shape)
            '''
        weight = weight.cpu().detach().numpy()
        mean = np.mean(weight)
        min_val = mean +2 * np.std(weight)
        max_val = mean -2 * np.std(weight)
        for x,text in [(weight[:,:weight.shape[1]//2], "rg"),(weight[:,weight.shape[1]//2:], "lbp")]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(x, vmin =min_val, vmax=max_val)
            fig.colorbar(cax)
            self.writer.add_figure("weights/"+text, fig, global_step=str(step))
            self.add_data("weights_sum/"+text, np.sum(np.abs(x)), step)  

    def log_weights(self, model, epoch):
        return self.log_weights2(model, epoch)
        if self.debug:
            # Histograms and distributions of network parameters
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                #print(tag, torch.min(value.data), torch.max(value.data))
                self.writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
                # second check required for buffers that appear in the parameters dict but don't receive gradients
                if value.requires_grad and value.grad is not None:
                    #print(tag+'grad', torch.min(value.grad.data), torch.max(value.grad.data))
                    self.writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")
    
    def log_weights2(self, model, epoch):
        if self.debug:
            # Histograms and distributions of network parameters
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                #print(tag, torch.min(value.data), torch.max(value.data))
                self.writer.add_scalar("max_"+tag, torch.max(value.data.cpu()), epoch)   
                self.writer.add_scalar("min_"+tag, torch.max(value.data.cpu()), epoch)  

                # second check required for buffers that appear in the parameters dict but don't receive gradients
                if value.requires_grad and value.grad is not None:
                    #print(tag+'grad', torch.min(value.grad.data), torch.max(value.grad.data))
                    self.writer.add_scalar("max_"+tag+ '/grad', torch.max(value.grad.data.cpu()), epoch)   
                    self.writer.add_scalar("min_"+tag+ '/grad', torch.max(value.grad.data.cpu()), epoch)  


    def log_confusion(self, step, matrix, name):
        """
        Visualization of confusion matrix. Is saved to hard-drive and TensorBoard.
        Parameters:
            step (int): Counter usually specifying steps/epochs/time.
            matrix (numpy.array): Square-shaped array of size class x class.
                Should specify cross-class accuracies/confusion in percent
                values (range 0-1)
        """

        #all_categories = sorted(class_dict, key=class_dict.get)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        # Set up axes
        ax.set_xticklabels([''] + list(range(matrix.shape[0])), rotation=90)#ax.set_xticklabels([''] + all_categories, rotation=90)
        ax.set_yticklabels([''] + list(range(matrix.shape[0])))#ax.set_yticklabels([''] + all_categories)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # Turn off the grid for this plot
        ax.grid(False)
        plt.tight_layout()

        self.writer.add_figure("Confusion/"+name, fig, global_step=str(step))
        #plt.savefig(os.path.join(save_path, 'confusion_epoch_' + str(step) + '.png'), bbox_inches='tight')

    def log_image_grid(self, images, epoch, name, save_path=None, channelwise = True):
        size = images.shape
        if channelwise:
            imgs_viz = torch.zeros([size[0]*size[1],1,size[2],size[3]])
            for c in range(size[1]):
                imgs_viz[c*size[0]:(c+1)*size[0],0]=images[:,c]

            imgs = torchvision.utils.make_grid(imgs_viz, nrow=size[0], padding=5)
        else:
            imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size[0])), padding=5)

        if save_path:
            torchvision.utils.save_image(imgs_viz, os.path.join(save_path, name + '_epoch_' + str(epoch + 1) + '.png'),
                                        nrow=int(math.sqrt(size)), padding=5)
        self.writer.add_image(name, imgs, epoch)

    def log_image_grid_errors(self, images, rows, epoch, name, save_path=None):
        imgs = torchvision.utils.make_grid(images, nrow= rows, padding=5)
        if save_path:
            torchvision.utils.save_image(images, os.path.join(save_path, name + '_epoch_' + str(epoch + 1) + '.png'),
                                        nrow=rows, padding=5)
        self.writer.add_image(name, imgs, epoch)

    def log_args(self,args):
        txt = ""
        for a in vars(args):
            txt += "%s: %s \n"%(a, str(getattr(args, a)))
        self.add_txt(txt, 'args')

    def vizualize_rg_dataset(self, dataset, num_images =8):
        def gray_to_RGB(x):
            return torch.cat([x,x,x], dim =0)
        #dataset_image
        trans = transforms.Compose([ResizeNP(224,224),transforms.ToTensor()])
        subdataset = torch.utils.data.Subset(dataset.imageset_train,range(num_images))
        grid_images = [trans(img) for (img,m) in  subdataset]

        #mask image
        grid_images += [trans(m) for (img,m) in  subdataset]

        #rgConf
        subdataset_preprocessed = torch.utils.data.Subset(dataset.trainset,range(num_images))
        grid_images += [gray_to_RGB(img[:1]) for (img,m) in subdataset_preprocessed] #r
        grid_images += [gray_to_RGB(img[1:2]) for (img,m) in subdataset_preprocessed] #g
        grid_images += [gray_to_RGB(img[2:3]) for (img,m) in  subdataset_preprocessed] #conf

        images = torchvision.utils.make_grid(grid_images, nrow = num_images)
        self.writer.add_image("imageset", images, 0)
    
    def vizualize_rg_lbp_nn_pipeline(self, dataset,  model, epoch, num_images =8, rg =False, rgConf = False, lbp =False, lbpConf =False):
        grid_images = []
        def gray_to_RGB(x):
            return torch.cat([x,x,x], dim =0)
        # dataset_image
        np.random.seed(0)
        subdataset = torch.utils.data.Subset(dataset.valset,np.random.randint(len(dataset.valset), size=num_images))
        
        if False:
            # RECONSTRUCTION TARGET
            grid_images += [l[1] for (img,l) in  subdataset if torch.is_tensor(l[1])]

        # INPUT
        if not rg and not lbp:
            grid_images += [img for (img,_) in  subdataset]

        # rgConf
        if rg:
            grid_images += [gray_to_RGB(img[:1]) for (img,m) in subdataset] #r
            grid_images += [gray_to_RGB(img[1:2]) for (img,m) in subdataset] #g
            if rgConf:
                grid_images += [gray_to_RGB(img[2:3]) for (img,m) in  subdataset] #conf1
                grid_images += [gray_to_RGB(img[3:4]) for (img,m) in  subdataset]

        # LBP
        if lbp:
            pre_channels = (4 if rgConf else 2)if rg else 0
            grid_images += [gray_to_RGB(img[pre_channels:pre_channels+1]) for (img,m) in subdataset] #lbp1
            grid_images += [gray_to_RGB(img[pre_channels+1:pre_channels+2]) for (img,m) in subdataset] #lbp2
            grid_images += [gray_to_RGB(img[pre_channels+2:pre_channels+3]) for (img,m) in  subdataset] #lbp3
            if lbpConf:
                grid_images += [gray_to_RGB(img[pre_channels+3:pre_channels+4]) for (img,m) in subdataset] #conf1
                grid_images += [gray_to_RGB(img[pre_channels+4:pre_channels+5]) for (img,m) in subdataset] #conf2
                grid_images += [gray_to_RGB(img[pre_channels+5:pre_channels+6]) for (img,m) in  subdataset] #conf3


        model.train(False)
        data = torch.stack([data for data,_ in subdataset]).to(device) #create batch
        
        # RECONSTRUCTION
        if False:
            out = model(data)[1][0]
            if torch.is_tensor(out):
                grid_images += [nn.Sigmoid()(out[i]).to("cpu") for i in range(out.shape[0])]

        images = torchvision.utils.make_grid(grid_images, nrow = num_images)
        self.writer.add_image("imageset", images, epoch)


class DefaultLogger():# prints only to console, no state dict etc
    def __init__(self):
        pass

    def add(self, text):
        print("log:", text)
    
    def save_model(self, model):
        pass