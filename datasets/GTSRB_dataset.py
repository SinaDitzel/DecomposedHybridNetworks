import torch
import torchvision
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from models.numpy_image_transforms import CropRoi, CropRoi_extra5, GaussianNoise, ResizeNP, RandomCropNP, RandomRotateNP, ToFloatTensor

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import random
import scipy
#from skimage import color, exposure, transform

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def load_image_np(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return np.asarray(img)

NO_ROI = 0
APPLY_ROI = 1
RETURN_ROI = 2
class Roi_image_loader():
    def __init__(self, path, roi_mode = NO_ROI):
        self.rois = {}
        self.roi_mode = roi_mode
        if os.path.isdir(path):
        #if (path).is_dir():
            root = path
            classes = [d.name for d in os.scandir(root) if d.is_dir()]
            for c in classes:
                csv_path = root +c+"/GT-"+c+".csv"
                self.read_rois_from_csv(root+c+"/", csv_path)
        elif path.endswith(".csv"):
            splited_path = path.split('/')
            root = path[:-len(splited_path[-1])]
            self.read_rois_from_csv(root, path)
       
    def read_rois_from_csv(self,root, path, verbose = False):
        data = pd.read_csv(path, sep =";")
        for i in range(data.shape[0]):
            filename = data['Filename'][i]
            rois = (data['Roi.X1'][i],data['Roi.Y1'][i],data['Roi.X2'][i],data['Roi.Y2'][i])
            self.rois.update({root+filename: rois })
            if verbose:
                print(root+filename)

    def __call__(self, path):
        with open(path, 'rb') as f:
            img = load_image_np(f)
            if self.roi_mode == NO_ROI:
                return img
            crop_area = self.rois[path]  
            if self.roi_mode == APPLY_ROI:
                return img[crop_area]        
            return img,crop_area



RED_c= [0,1,2,3,4,5,7,8,9,10,15,16] #Circle with red margin
RED_cf = [14,17] #Filled Red 
BLUE_cf = [33,34,35,36,37,38,39,40] # Blue filled circle
RED_t = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #Triangle Red Surrounding upword
RED_t2 = [13] #Triangle red(vorfahrt)
GRAY = [6,32,41,42] #Gray 
YELLOW = [12] #yellow rectangle(vorfahrt)


class SortedByColoredShape():
    def __init__(self):
        self.map = self.create_map()
        self.num_classes = 43
        self.class_names=["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
                    "Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)",
                    "End of speed limit (80km/h)","Speed limit (100km/h) ", "Speed limit (120km/h)",
                    "No passing", "No passing veh over 3.5 tons", "Right-of-way at intersection",
                    "Priority road", "Yield", "Stop", "No vehicles" , "Veh > 3.5 tons prohibited ",
                    "No entry", "General caution", "Dangerous curve left" ,"Dangerous curve right",
                    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
                    "Traffic signals" ,"Pedestrians", "Children crossing ", "Bicycles crossing", 
                    "Beware of ice/snow", "Wild animals crossing ", "End speed + passing limits ", 
                    "Turn right ahead ", "Turn left ahead ", "Ahead only", "Go straight or right",
                    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
                    "End of no passing ", "End no passing veh > 3.5 tons" ]
        inv_map = {v: k for k, v in self.map.items()}
        self.class_names = [self.class_names[i[1]] for i in inv_map.items()]
    
    def create_map(self):
        idx = 0
        m ={}
        for group in [RED_c, RED_cf, BLUE_cf, RED_t, RED_t2, GRAY, YELLOW]:
            for i in group:
                m.update({i:idx})
                idx += 1
        return m

    def __call__(self,x):
        return self.map[x]

class DatasetFolderTest(DatasetFolder):
    def __init__(self, *args):
        super(DatasetFolderTest,self).__init__(*args)
    
    def __getitem__(self, index):
        print("getitem %i"%(index))
        result = super().__getitem__(index)
        print("finished getitem %i"%(index))
        return result

class ColorShapeClasses():
    def __init__(self):
        self.map = self.create_map()
        self.num_classes = 7
        self.class_names=["Circle with red margin","Filled Red","Blue filled circle","Triangle Red Surrounding upword",
                                "Triangle red(Yield)","Gray ","yellow rectangle(priority road)"]
    
    def create_map(self):
        idx = 0
        m ={}
        for group in [RED_c, RED_cf, BLUE_cf, RED_t, RED_t2, GRAY, YELLOW]:
            for i in group:
                m.update({i:idx})
            idx += 1
        return m

    def __call__(self,x):
        return self.map[x]

class KeepInputandRoi():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self,x):
        return np.concatenate([self.transform((x[0],x[1])), x[0]], axis=2),x[1]

class SplitTransform():
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self,x):
        return (self.transform1(x[0]), self.transform2(x[1]))

class ApplyOnBackChannels():
    def __init__(self, transform, channels):
        self.transform = transform
        self.channels = channels

    def __call__(self,x):
        return np.concatenate([x[:,:,:-self.channels],self.transform(x[:,:,-self.channels:])], axis=2)

class MyDatasetFolder(DatasetFolder):
    def __init__(self,*args, decompositions = [], inputAdditionalTagret = False, p=-1, p_seed=0, tag = None):
        self.decompositions = decompositions
        self.toTensor = transforms.Compose([transforms.ToTensor(), ToFloatTensor()])
        self.inputAdditionalTagret = inputAdditionalTagret
        super().__init__(*args)
        #self.samples = self.samples[:int(len(self.samples)*p)]
        if p >0: # only use portion of dataset
            samples_small = []
            targets_small = []
            num_per_class = p# int(p*len(self.samples)/len(self.classes))
            random.seed(p_seed)
            for i in range(len(self.classes)):
                inputs_tmp = []
                for j, inp in enumerate(self.samples):
                    if self.targets[j]== i: 
                        inputs_tmp.append(inp)
                print(F'Class {i}: original samples: {len(inputs_tmp)}')
                c = random.sample(range(len(inputs_tmp)), min(num_per_class, len(inputs_tmp)))
                samples_small += [inputs_tmp[i] for i in c]
                targets_small += [i for _ in range(min(num_per_class, len(inputs_tmp)))]
            print(F"Using only portion of the datset(old size: {len(self.samples)}), new size:{len(samples_small)}")
            self.samples = samples_small
            self.targets = targets_small
        self.tag = tag
    
    def __getitem__(self, index: int):
        #getitem with load tag

        original_sample, label = super().__getitem__(index)
        sample = original_sample
        if len(self.decompositions)>0:
            transformed_sample = []
            for op in self.decompositions:
                if 'Conf' in str(op) and self.tag is not None:
                    filename, _ = self.samples[index]
                    split_path = filename.split('/GTSRB/')
                    filepath = os.path.join(split_path[0],'GTSRB/', self.tag, str(op),  split_path[1].split('.')[0]+'.npy')
                    if os.path.isfile(filepath):
                        tmp = np.load(filepath)
                    else:
                        tmp = op(sample)
                        os.makedirs(os.path.dirname(filepath), exist_ok = True)
                        np.save(filepath, tmp)
                else:
                    tmp = op(sample)
                #print(str(op), tmp.shape)
                transformed_sample.append(tmp)
            sample = np.concatenate(transformed_sample, axis=2)
        if self.inputAdditionalTagret:
            return self.toTensor(sample) ,(label, self.toTensor(original_sample))
        else:
            return self.toTensor(sample), label
    

class DatasetwithCSV(Dataset):
    def __init__(self, root_dir, csv_file,transform = None, target_transform=None, roi_mode = NO_ROI, 
                decompositions =[], inputAdditionalTagret = False, tag=None):
        self.annotations = pd.read_csv(csv_file, sep = ";")
        self.Roi_img_loader = Roi_image_loader(root_dir+"GT-final_test.test.csv", roi_mode)
        self.roi_mode = roi_mode
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.toTensor = transforms.Compose([transforms.ToTensor(), ToFloatTensor()])
        self.decompositions = decompositions
        self.inputAdditionalTagret = inputAdditionalTagret
        self.tag = tag

    def __getitem__(self, i):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
        image = self.Roi_img_loader(img_path)
        label = self.annotations.iloc[i, 7]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = image

        if len(self.decompositions)>0:
            transformed_sample = []
            for op in self.decompositions:
                if 'Conf' in str(op) and self.tag is not None:
                    split_path = img_path.split('/GTSRB/')
                    filepath = os.path.join(split_path[0],'GTSRB/', self.tag, str(op),  split_path[1].split('.')[0]+'.npy')
                    if os.path.isfile(filepath):
                        tmp = np.load(filepath)
                    else:
                        tmp = op(sample)
                        os.makedirs(os.path.dirname(filepath), exist_ok = True)
                        np.save(filepath, tmp)
                else:
                    tmp = op(sample)
                #print(str(op), tmp.shape)
                transformed_sample.append(tmp)
            sample = np.concatenate(transformed_sample, axis=2)
        if self.inputAdditionalTagret:
            return self.toTensor(sample) ,(label, self.toTensor(image))
        else:
            return self.toTensor(sample), label

    def __len__(self):
        return len(self.annotations)


class GTSRB():#German Traffic Sign Recognition Benchmark
    def __init__(self, root, batch_size=64, num_workers = 4, less_classes= False, 
                roi_mode= RETURN_ROI, decompositions=[], reconstruction = False,  unsup_train=False, p =-1,  p_seed=0, random_rotate=True,
                input_size=128):
        self.batch_size = batch_size
        self.path = root
        self.train_root = self.path+"GTSRB_Final_Training/Images_train/"
        self.val_root = self.path+"GTSRB_Final_Training/Images_val/"
        self.test_root = self.path+"GTSRB_Final_Test/Images/"
        test_csv_file =  self.path+ "GTSRB_Final_Test/GT-final-test.csv"
        self.name = "GTSRB"
        self.reconstruction = reconstruction
        self.unsup_train = unsup_train

        if less_classes:
            target_transform = ColorShapeClasses()
        else:
            target_transform = SortedByColoredShape()
            #resize = ResizeNP(self.patch_size,self.patch_size)
            #gt_images =[resize(load_image_np(os.path.join(self.path,"gt_signs/sign_%i.png"%i))) for i in range(self.num_classes)]
            #self.gt_images = [gt_images[it[1]] for it in inv_map.items()]

        #DATASETS with preprocessed images
        # preprocessing_train = transforms.Compose(
        #                         [CropRoi_extra5(),
        #                         ResizeNP(53,53)]
        #                         +([RandomRotateNP(15)] if  random_rotate else [])
        #                         +[RandomCropNP(48,48)]) #+([GaussianNoise(sigma)] if sigma is not None else [])
        #print("preprocessing train:", preprocessing_train)      
        resize_size = (input_size, input_size)          
        preprocessing_train= preprocessing_eval = transforms.Compose(
                                [CropRoi(),
                                ResizeNP(resize_size[0],resize_size[1])])#+([GaussianNoise(sigma)] if sigma is not None else [])
        tag = F'GTSRB_preprocessed_roi_resize({resize_size})'# change depending on preprocessing train
        self.num_classes = target_transform.num_classes
        self.class_names = target_transform.class_names
        print('GTSRB num classes', target_transform.num_classes)
        #IMAGESETS
        self.trainset = MyDatasetFolder(self.train_root, 
                                        Roi_image_loader(self.train_root, roi_mode), IMG_EXTENSIONS, 
                                        preprocessing_train, target_transform, 
                                        decompositions = decompositions,
                                        inputAdditionalTagret=(self.reconstruction or self.unsup_train), p=p, p_seed=p_seed, tag=tag)
        self.valset = MyDatasetFolder(self.val_root, 
                                        Roi_image_loader(self.val_root, roi_mode), IMG_EXTENSIONS, 
                                        preprocessing_eval, target_transform, 
                                        decompositions = decompositions,
                                        inputAdditionalTagret=(self.reconstruction or self.unsup_train), tag= tag)
        self.testset = DatasetwithCSV(root_dir = self.test_root,
                                    csv_file =  test_csv_file,
                                    transform = preprocessing_eval,
                                    target_transform=target_transform,
                                    roi_mode = roi_mode,
                                    decompositions = decompositions,
                                    inputAdditionalTagret=(self.reconstruction or self.unsup_train), tag=tag)
        #DATALOADER
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=num_workers)#, timeout = (150 if num_workers>0 else 0))
    
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=num_workers)#,timeout = (150 if num_workers>0 else 0))

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=num_workers)#,timeout = (150 if num_workers>0 else 0))