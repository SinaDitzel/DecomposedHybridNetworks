import math
import sys
import numpy as np
from scipy import stats

from models.gaussianNoiseEstimator import GaussianNoiseEstimator
from models.numpy_image_transforms import Intensity
from models.chi_square_confidence import Confidence

from device import device


#input np.array w,h,c

class LBP(object):
    '''create p neighbouring positions in r distance to center pixel
       threshholding neighboring pixel with center pixel
       if neighbour >= center pixel => 1 else 0
       create decimal number from binary result'''
    def __init__(self, radius=1, points=8, conf=False, no_decimal=False, 
                 lambdas=(lambda d:[0, 100, 1000, np.median(d)*0.5]), weights=None, prior=0.5):
        self.radius = radius
        self.points = points
        self.withConf = conf
        if self.withConf:
            self.noiseEstimator = GaussianNoiseEstimator()
            lambdas = lambdas#lambda d :[0, 1, 10, np.median(d)*0.5]
            if weights == None:
                weights = [1/len(lambdas(0)) for i in lambdas(0)]
            else:
                weights = [0.25,0.25,0.25,0.25]
            #print(F'LBP with lambdas(1000):{lambdas(1000)}, weights {weights}')
            self.conf = Confidence(lambdas, weights, k=self.points, prior_H0=prior)
        self.no_decimal = no_decimal

    def positions(self):
        angle = 2 * math.pi / self.points
        return [(int(round(self.radius * math.cos(angle*i))),
                 int(round(self.radius * math.sin(angle*i))))
                 for i in range(self.points)]

    def noise_prop(self, sigma):
        #Calculate covariance matrix cov_matrix
        cov_matrix = np.full((self.points,self.points), sigma**2) #self.points gives dimenion of binary number
        np.fill_diagonal(cov_matrix,2*sigma**2)
        return cov_matrix

    def mahalanobis_distance_pixel(self, point, mean, inv_cov_matrix):
        #print (inv_cov_matrix)
        return np.matmul(np.matmul(np.transpose(point-mean),inv_cov_matrix),(point-mean))

    def get_inv_cov_matrix(self, d, sigma):
        inv_cov_matrix = np.full((d,d), -1/((d+1)*sigma**2)) #self.points gives dimenion of binary number
        np.fill_diagonal(inv_cov_matrix, d/((d+1)*sigma**2))
        return inv_cov_matrix

    def lbp_conf(self, intensity, diffs):
        if False:
            return np.ones((diffs.shape[0], diffs.shape[1],1))
        mah = np.zeros((diffs.shape[0], diffs.shape[1]))
        sigma = self.noiseEstimator(intensity, intensity)+1e-7
        #print(sigma)
        inv_cov_matrix = self.get_inv_cov_matrix(self.points, sigma)
        for i in range(diffs.shape[0]):
            for j in range(diffs.shape[1]):
                mah[i,j] = self.mahalanobis_distance_pixel(diffs[i,j].reshape(self.points,1), np.zeros((self.points,1)), inv_cov_matrix)
        return self.conf(mah)

    def __call__(self, data):
        #compare the the neighboring pixels to that of the central pixel
        neigbour = np.zeros((data.shape[0], data.shape[1], self.points))
        max_x, max_y = data.shape[1], data.shape[0]
        for i, (y,x) in enumerate(self.positions()):
            neigbour[max(0,-y):min(max_y, max_y-y),
                     max(0,-x):min(max_x, max_x-x),
                     i] = data[max(0,y):min(max_y, max_y+y),
                               max(0,x):min(max_x, max_x+x)]
        diff = neigbour - np.repeat(data[:, :, np.newaxis], self.points, axis=2) #height,width, num_points

        if self.withConf:
            c = self.lbp_conf(data, diff)

        #convert to binary: 0 if lessequall
        diff[diff >= 0] = 1
        diff[diff < 0] = 0
        if self.no_decimal:
            return diff

        #convert to decimal
        for i in range(self.points):
            diff[:,:,i] = diff[:,:,i] * 2**i
        d = np.sum(diff, axis=2)
        out = d/float(2**(self.points+1)-1)#max decimal number, defined by number of points
        if self.withConf:
            out = np.concatenate([np.expand_dims(out, axis=2), c], axis = 2)
        return out

    def __str__(self):
        return "LBP(radius_%i_points_%i)"%(self.radius, self.points)


lambda_values = [lambda d: [0],
                     lambda d: [np.median(d) * 0.5],
                     lambda d: [np.median(d)],
                     lambda d: [0, np.median(d) * 0.5],
                     lambda d: [0, 10, 100, np.median(d) * 0.5],
                     lambda d: [0, 100, 1000, np.median(d) * 0.5]]
class MultipleLBP():
    def __init__(self,parameters, conf=False, lambdas=(lambda d:[0, 100, 1000, np.median(d)*0.5]), priors = None, no_decimal=False):
        self.lbps =[]
        self.conf = conf
        self.parameters= parameters
        
        if priors is None:
            priors = [0.5 for i in parameters]
        self.priors = priors
        for (r, p), prior in zip(parameters, priors):
            self.lbps.append(LBP(r,p, conf, no_decimal, lambdas, prior=prior))
        self.channels = len(self.lbps) if not no_decimal else sum([l.points for l in self.lbps])
        
        self.intensity = Intensity()

        if conf:
            self.forward = self.forward_with_conf
        else:
            self.forward = self.forward_without_conf
        self.no_decimal = no_decimal

    def forward_without_conf(self, x):
        x = self.intensity(x) # convert to grayscale
        out = np.zeros((x.shape[0], x.shape[1], self.channels))
        c_it = 0
        for t in self.lbps:
            out_tmp = t(x)
            #TODO small bug. out_tmp has shape (h,w) not (h,w,1)
            #c_tmp = c_it+ out_tmp.shape[2]
            out[:,:,c_it]= out_tmp
            c_it +=1 #c_tmp
        return out

    def forward_with_conf(self, x):
        #convert to grayscale
        x = self.intensity(x)
        out = np.zeros((x.shape[0], x.shape[1], 2*len(self.lbps)))
        for i,t in enumerate(self.lbps):
            li = t(x)
            out[:,:,i] = li[:,:,0]
            out[:,:,len(self.lbps)+i] = li[:,:,1]
        #print("done")
        return out

    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return F"LBP({self.parameters})"+(F"_Conf(p= {[int(p*100) for p in self.priors]}%)" if self.conf else "")