import matplotlib.pyplot as plt
import matplotlib.colors as colors

from models.lbp import LBP,MultipleLBP
from models.rg import NormalizedRG
import torchvision.transforms as transforms
import numpy as np
import torch
from datasets.GTSRB_dataset import GTSRB
from tqdm.auto import tqdm
from torchvision.transforms import transforms

from priors import P_H0_rg_GTSRB, P_H0_LBP_5_40_GTSRB,P_H0_LBP_1_8_GTSRB,P_H0_LBP_3_24_GTSRB

def lbp_lambda(x):
    #lbp_transform = LBP(radius=3, points=24)
    lbp_transform = MultipleLBP([(1, 8), (3, 24), (5, 40)], conf=False, no_decimal=False,
                            priors=[P_H0_LBP_1_8_GTSRB, P_H0_LBP_3_24_GTSRB, P_H0_LBP_5_40_GTSRB]
                            )

    all_imgs = []
    transform = transforms.Grayscale()
    for i in range(x.shape[0]):
        current_img = x[i].detach().cpu().numpy()
        current_img = current_img.transpose(1, 2, 0)
        current_img = lbp_transform(current_img)
        all_imgs.append(current_img)
    all_imgs = np.asarray(all_imgs)
    img_out = torch.Tensor(all_imgs)

    return img_out

def lbpConf_lambda(x):
    x = x * 255
    x = x.type(torch.int64)
    lbp_transform = MultipleLBP([(1, 8), (3, 24), (5, 40)], conf=True, no_decimal=False,
                            priors=[P_H0_LBP_1_8_GTSRB, P_H0_LBP_3_24_GTSRB, P_H0_LBP_5_40_GTSRB]
                            )
    all_imgs = []
    transform = transforms.Grayscale()
    for i in range(x.shape[0]):
        #current_img_gray = transform(x[i])
        current_img = x[i].detach().cpu().numpy()
        current_img = current_img.transpose(1,2,0)
        current_img = lbp_transform(current_img)
        all_imgs.append(current_img)
    all_imgs = np.asarray(all_imgs)
    img_out = torch.Tensor(all_imgs)
    return img_out

def rg_lambda(x):
    x = x*255
    x= x.type(torch.int64)
    rg_norm = NormalizedRG(conf=False,p_H0=P_H0_rg_GTSRB)
    #print('shape i rg_lambda',x.shape)
    all_imgs = []
    for i in range(x.shape[0]):
      current_img= x[i].permute(1, 2, 0).detach().cpu().numpy()
      all_imgs.append(rg_norm(current_img))
    all_imgs = np.asarray(all_imgs)
    #print("shape all images",all_imgs.shape)
    img_out = torch.Tensor(all_imgs).permute(0,3,1,2)
    #print("shape all images final",all_imgs.shape)

    return img_out

def rgConf_lambda(x):
    x = x * 255
    x = x.type(torch.int64)
    rg_norm = NormalizedRG(conf=False,p_H0=P_H0_rg_GTSRB)
    #print('shape i rg_lambda',x.shape)
    all_imgs = []
    for i in range(x.shape[0]):
      current_img= x[i].permute(1, 2, 0).detach().cpu().numpy()
      all_imgs.append(rg_norm(current_img))
    all_imgs = np.asarray(all_imgs)
    #print("shape all images",all_imgs.shape)
    img_out = torch.Tensor(all_imgs).permute(0,3,1,2)
    #print("shape all images final",all_imgs.shape)

    return img_out


def plot_images(images, colorbar = False, RGB = False, own_scale = False,  size=(5.5,4), same_size = False, lognorm = False):
    fig, axs = plt.subplots(1, len(images), figsize=(size[0]*len(images), size[1]), sharex=same_size, sharey=same_size)
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for img, ax in zip(images, ([axs] if len(images)==1 else axs.ravel())):
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
    plt.show()