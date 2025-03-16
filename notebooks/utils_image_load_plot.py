#Dependencies
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  
import matplotlib.colors as colors

def plot_images(images, colorbar = False, RGB = False, own_scale = False,  size=(5.5,4), same_size = False, lognorm = False):
    fig, axs = plt.subplots(1, len(images), figsize=(size[0]*len(images), size[1]), sharex=same_size, sharey=same_size, squeeze=True)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0)
    #fig, axs = pyplot.subplots(nrows=1, ncols=4, gridspec_kw={'wspace':0, 'hspace':0},squeeze=True)

    for img, ax in zip(images, ([axs] if len(images)==1 else axs.ravel())):
        ax.axis("off")
        if RGB:
            im = ax.imshow(img)
        else:
            if not own_scale:
                im = ax.imshow(img, cmap="gray", vmin = 0, vmax = 1)
            else:
                if lognorm:
                    eps = 1e-10
                    norm = colors.LogNorm(vmin=max(eps,img.min()), vmax=img.max())
                    #norm = colors.PowerNorm(4)
                    im = ax.imshow(img, cmap="gray", norm = norm)
                else:
                    im = ax.imshow(img, cmap="gray", vmin = 0, vmax = np.max(img))
                    
                if colorbar:
                    fig.colorbar(im, ax=ax)
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()
    
def plot_hists(images,range = None,  size=(5.5,4), bins='sqrt', cum =False, horiz = True, log =False, max_plot_values = 2500):
    if horiz:
        fig, axs = plt.subplots(1, len(images), figsize=(size[0]*len(images), size[1]))
    else:
        fig, axs = plt.subplots(len(images),1 , figsize=(size[1],size[0]*len(images)))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    '''if range == None:
        max_range = 0
        for i in images:
            if i.max()>max_range:
                max_range = i.max()
        range =(0,max_range)'''
    for img, ax in zip(images, ([axs] if len(images)==1 else axs.ravel())):
        if len(img) > max_plot_values:
            img = img[np.random.choice(len(img), size=max_plot_values)]
        im = ax.hist(img, bins = bins, range = range, log = log)
        if cum :
            ax.hist(img, bins = bins, range = range, cumulative=True, histtype='step', log = log)
    plt.show()   
    

def plot_distributions(models, ranges, markers=None, size=(5.5,4), same_size = False):
    if markers is None:
        markers = [[]]*len(models)
    fig, axs = plt.subplots(1, len(models), figsize=(size[0]*len(models), size[1]), sharex=same_size, sharey=same_size)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for m,r, l, ax in zip(models,ranges, markers, ([axs] if len(models)==1 else axs.ravel())):
        for pdf in m:
            x = np.linspace(r[0], r[1], 1000)
            ax.plot(x, pdf(x).reshape(-1))
        for line in l:
            ax.axvline(x=line[0], c="y", label = line[1])
    plt.show()

SIGN_SPEED20 = ("00000","00005_00000.ppm")
SIGN_SPEED20_1 = ("00000","00004_00015.ppm")
SIGN_YELLOW = ("00012", "00006_00027.ppm")
SIGN_BLUE = "00034", "00008_00026.ppm"
SIGN_TRAFFICLIGHT = "00026" , "00001_00027.ppm"
#SIGN_GRAY =("00041","00003_00002.ppm")
SIGN_GRAY =("00042","00004_00026.ppm")
SIGN_STOP =("00014","00004_00008.ppm") 
GTSRB_TEST_IMAGES = [SIGN_SPEED20, SIGN_SPEED20_1, SIGN_YELLOW, SIGN_BLUE, SIGN_TRAFFICLIGHT, SIGN_GRAY, SIGN_STOP]

def load_GTSRB_test_images(path = "/data/GTSRB/GTSRB_Final_Training/Images_train/"):
    images = []
    for folder, x in GTSRB_TEST_IMAGES:#os.listdir(path):#[im1, im4]:#os.listdir(path):#
        f = os.path.join(path, folder, x)
        #print(f)
        img = Image.open(f)
        img = img.convert('RGB')
        img = np.asarray(img)
        images.append(img)
    return images


def get_rois_GTSRB_test_images(path="/data/GTSRB/GTSRB_Final_Training/Images_train/"):
    rois=[]
    for folder, x in GTSRB_TEST_IMAGES:
        csv_path = os.path.join(path, folder, 'GT-'+folder+'.csv')
        import pandas as pd
        data = pd.read_csv(csv_path, sep =";")
        for i in range(data.shape[0]):
            filename = data['Filename'][i]
            if data['Filename'][i]== x:
                rois.append((data['Roi.X1'][i],data['Roi.Y1'][i],data['Roi.X2'][i],data['Roi.Y2'][i]))
    return rois


"""
import math
import copy
import sys
import torchvision.transforms.functional as F
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser("lbp_noise_prop.py"))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import pandas as pd
import scipy
from scipy import ndimage, misc, signal, integrate, stats, optimize
from skimage import feature
"""