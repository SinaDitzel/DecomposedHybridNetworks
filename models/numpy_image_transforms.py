
import cv2
import numpy as np
import scipy
import random

class Intensity():
    def __call__(self, x):
        return x.sum(axis =2)//3

class Saturation():
    def __call__(self,x):
        i = x.sum(axis =2)
        min_c = (np.minimum(np.minimum(x[:,:,0], x[:,:,1]), x[:,:,2]))
        return 1 - np.divide(min_c, i, out= np.zeros(min_c.shape), where=i!=0)

class GaussianNoise():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self,x):
        return np.clip(x + np.random.normal(0, self.sigma, x.shape),0, 255).astype(np.uint8)
        
class ResizeNP_smallside():
    def __init__(self, size, list =False):
        self.size =size

    def __call__(self, img):
        if img.shape[0]< img.shape[1]:
            size_x = self.size
            size_y = int(img.shape[1]*(size_x/img.shape[0]))
        else:
            size_y = self.size
            size_x = int(img.shape[0]*(size_y/img.shape[1]))
        return cv2.resize(img, dsize=(size_y, size_x), interpolation=cv2.INTER_LINEAR)

class PIL2NP():
    def __call__(self, img):
        return np.asarray(img)

class ResizeNP():
    def __init__(self, size_x, size_y, list =False):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        img= cv2.resize(img, dsize=(self.size_y, self.size_x), interpolation=cv2.INTER_LINEAR)
        return img

class RandomRotateNP():
    def __init__(self, degree):
        self.d = degree

    def __call__(self,img):
        d = random.uniform(0,self.d)
        return scipy.ndimage.rotate(img, d, reshape=False)

class CenterCropNP():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        x1 = max(0,int(img.shape[1]/2-self.size_x/2))
        x2 = min(int(img.shape[1]/2+self.size_x/2),img.shape[1])
        y1 = max(0,int(img.shape[0]/2-self.size_y/2))
        y2 = min(int(img.shape[0]/2+self.size_y/2),img.shape[0])
        return img[y1:y2,x1:x2]

class RandomHorizontalFlipNP():
    def __call__(self,img):
        if random.randint(0,1):
            return cv2.flip(img, 1) 
        return img

class RandomCropNP():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def __call__(self, img):
        start_x = random.randint(0,(img.shape[1]-self.size_x))
        start_y = random.randint(0,(img.shape[0]-self.size_y))
        return img[start_y:start_y+self.size_y,start_x:start_x+self.size_x,:]

class ToFloatTensor():
    def __call__(self,x):
        return x.float()

class CropRoi_extra5():
    def __call__(self,x):
        img, (x1,y1,x2,y2) = x
        width, height = x2-x1, y2-y1
        '''#add 10 %
        x1 = min(0,x1 - 0.05 * width)
        x2 = max(img.size[0], x2 + 0.05 * width)
        y1 = min(0,y1 - 0.05 * height)
        y2 = max(img.size[1], y2 + 0.05 * height)
        '''
        img, (x1,y1,x2,y2) = x
        width, height = x2-x1, y2-y1
        #add 10 %
        x1 = min(0, int(x1 - 0.05 * width))
        x2 = max(img.shape[0], int(x2 + 0.05 * width))
        y1 = min(0,int(y1 - 0.05 * height))
        y2 = max(img.shape[1], int(y2 + 0.05 * height))
        #print(x1,x2,y1,y2)
        #return img.crop((x1,y1,x2,y2))
        return img[y1:y2, x1:x2]


class CropRoi():
    def __call__(self,x):
        img, (x1,y1,x2,y2) = x
        #return img.crop((x1,y1,x2,y2))
        return img[y1:y2, x1:x2]

class Histogram_Equalization_HSVNP():

    def __call__(self, img):
        #img = np.array(img)
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
        #img = Image.fromarray((img*255).astype(np.uint8))
        #opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #cv2.imshow('hsv_equalized',opencvImage)
        #cv2.waitKey(3000)
        return img

    def __str__(self):
        return "Histogram Equalization(based on HSV)"
