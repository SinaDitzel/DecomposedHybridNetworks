import numpy as np 
import warnings
import sys

from scipy import ndimage
from skimage import feature

from .numpy_image_transforms import Intensity, Saturation
from .gaussianNoiseEstimator import GaussianNoiseEstimator
from .chi_square_confidence import Confidence


class NormalizedRG():
    def __init__(self, conf = False, p_H0=0.5, lambdas=(lambda d:[0, 100, 1000, np.median(d)*0.5]), weights=None, randomConf = False):
        self.conf = conf
        self.channels = 2
        self.randomconf = randomConf # only for testing
        self.prior = p_H0
        if self.conf:
            self.noiseEstimator = GaussianNoiseEstimator()
            self.intensity = Intensity()
            self.saturation = Saturation()
            lambdas = lambdas#lambda d :[0, 10, 100, np.median(d)*0.5]
            if weights == None:
                weights = [1/len(lambdas(0)) for i in lambdas(0)]
            else:
                weights = [0.25,0.25,0.25,0.25]
            #print(F'rg with lambdas(1000):{lambdas(1000)}, weights {weights}')
            self.conf = Confidence(lambdas, weights, 2, p_H0)

    def __call__(self, x):
        rg = self.normalized_rg(x)
        out = rg = self.color_constancy_correction_rg(rg)
        img_cc = self.rg2RGB(rg, x.sum(axis=2, keepdims=True))
        if self.conf:
            if self.randomconf:
                c = np.random.rand(x.shape[0], x.shape[1])
            else:
                c = self.color_confidence_estimation(rg, x)
                if np.isnan(np.concatenate([rg, c],axis=2)).any():
                    print("rg:", np.isnan(rg).any())
                    print("c:", np.isnan(c).any())

                    print("rg:", rg)
                    print("c:", c)

                    sys.exit()
            out = np.concatenate([rg, c], axis=2)
        return out

    def normalized_rg(self, x):
        intensity = x.sum(axis =2, keepdims = True)
        rgb = np.divide(x, intensity, out= np.full(x.shape,1/3), where=intensity!=0)
        return rgb[:,:,:2]

    def rg2RGB(self, rgimg, intensity):
        img = np.zeros((rgimg.shape[0],rgimg.shape[1],3))
        img[:,:,:2] = rgimg
        img[:,:,2] = 1- (rgimg[:,:,0]+rgimg[:,:,1])
        img = (img*intensity)
        img[img>255] = 255 #color correction "error" handling
        return img.astype(np.uint8)

    def find_mode(self, values, exp_mode, delta, prec=0.01):
        """ returns mode in [exp_mode-delta, exp_mode+delta] """
        # sort values and use only range +-delta
        values = np.sort(values.reshape(-1))
        values = values[np.logical_and(values > exp_mode - delta,values < exp_mode + delta)]  
        if len(values)==0:
            return 1/3  
        # create histogram and smooth
        hist = np.histogram(values, max(1, int((np.max(values)-np.min(values))/prec)))
        smoothed_hist = ndimage.gaussian_filter1d(hist[0], sigma=0.1)
        # get argmax
        mode = hist[1][np.argmax(smoothed_hist)]
        return mode

    def find_mean(self, values, exp_mode, delta):
        """ returns mean in [exp_mode-delta, exp_mode+delta] """
        # sort values and use only range +-delta
        values = np.sort(values.reshape(-1))
        values = values[np.logical_and(values > exp_mode - delta,values < exp_mode + delta)]
        if len(values)==0:
            return 1/3      
        mode = np.mean(values)
        return mode

    def color_constancy_correction_rg(self, img_rg, max_shift=0.05, mean_as_mode=True):
        '''
        Color correction in the normalized rg space
            Assumption: 
                * the normalized color values have a mode at 1/3, 1/3 (white, gray, black areas)
                * the mode is only slightly shifted (parameter max_shift)
                * image can be corrrected by finding a factor that is muliplyed with all values
                such that the mode is then at (1/3 1/3)
            Method:
                * the mode of the normalized rg values closest to 1/3 1/3 is searched,
                * a corrective factor is cacluated to shift the values from the curent mode to 1/3            
        '''
        if mean_as_mode:
            r_mode = self.find_mean(img_rg[:,:,0], exp_mode=1/3, delta=max_shift)
            g_mode = self.find_mean(img_rg[:,:,1], exp_mode=1/3, delta=max_shift) 
        else:
            r_mode = self.find_mode(img_rg[:,:,0], exp_mode=1/3, delta=max_shift, prec=0.0005)# find current max in images
            g_mode = self.find_mode(img_rg[:,:,1], exp_mode=1/3, delta=max_shift, prec=0.0005) # find current max in images
            
        rg_c = np.zeros(img_rg.shape)
        rg_c[:,:,0] = img_rg[:,:,0] * 1/3 * 1/r_mode
        rg_c[:,:,1] = img_rg[:,:,1] * 1/3 * 1/g_mode

        # sifting could lead to too high values -> clip
        sum_rg = rg_c[:,:,0]+rg_c[:,:,1]
        rg_c[:,:,0][sum_rg > 1] = (rg_c[:,:,0] - ((sum_rg-1)/2))[sum_rg > 1] # sum should be max = 1
        rg_c[:,:,1][sum_rg > 1] = (rg_c[:,:,1] - ((sum_rg-1)/2))[sum_rg > 1] # sum should be max = 1
        return rg_c
 
    def color_confidence_estimation(self, rg,  img):
        intensity, saturation = self.intensity(img), self.saturation(img)
        noise = [self.noiseEstimator(img[:,:,i], intensity, saturation) for i in range(img.shape[2])] 
        d = self.mahalanobi_dist(rg, (1/3,1/3), img, noise)
        conf = self.conf(d)
        return np.repeat(conf, 2, axis=2) #same confidence for r and g

    def cov(self, img, sigma, r, g):
        var_R = np.square(sigma[0])
        var_G = np.square(sigma[1])
        var_B = np.square(sigma[2])

        R = img[:,:,0].astype(np.float64)
        G = img[:,:,1].astype(np.float64)
        B = img[:,:,2].astype(np.float64)
        eps = 1e-7
        S = R+G+B+eps
        var_I = ((var_R+var_G+var_B)/3)+eps

        var_r1 = (var_R/var_I)*(1-2*r)+3*r*r
        var_r = var_I/np.square(S)* var_r1

        var_g1 = (var_G/var_I)*(1-2*g)+3*g*g
        var_g = var_I/np.square(S)* var_g1
        
        cov_rg1= -((var_G/var_I)*r)-((var_R/var_I)*g)+3*r*g
        cov_rg = var_I/np.square(S)* cov_rg1 
        return np.array([[var_r,cov_rg],[cov_rg,var_g]])

    def mahalanobi_dist(self, rg_measurement, mean, img, sigma):
        eps = 1e-20
        r_mean,g_mean = mean
        cov_ma = self.cov(img, sigma, r_mean, g_mean)
        s_factor = 1/((cov_ma[0][0]*cov_ma[1][1])-np.square(cov_ma[0][1])+eps)
        s11 = s_factor * cov_ma[1][1]
        s12 = s_factor * - cov_ma[0][1]
        s22 = s_factor * cov_ma[0][0]
        r_diff = rg_measurement[:,:,0]-r_mean
        g_diff = rg_measurement[:,:,1]-g_mean
        mah_dist =  r_diff*r_diff*s11+2*r_diff*g_diff*s12+g_diff*g_diff*s22
        return mah_dist

    def __str__(self):
        return "rg"+(F"_Conf(p= {int(self.prior*100)}%)" if self.conf else "")

class PrintT():
    def __call__(self,x):
        print("result:",x)
        return x

