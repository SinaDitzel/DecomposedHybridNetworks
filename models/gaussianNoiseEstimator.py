
import numpy as np 
import matplotlib.pyplot as plt

from scipy import  signal

class GaussianNoiseEstimator():
    def __init__(self, p =0.1):
        self.p =p
        self.sobel1 = [[-1,-2, -1],
              [ 0, 0, 0],
              [ 1, 2, 1]]
        self.sobel2 = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]

        self.laplacian = [[1, -2, 1],#LaPlacioan Operator
            [-2, 4, -2],
            [1, -2, 1]]

    # returns a mask, for all pixels, 
    # if it belongs to p % of the positions with least of the edges
    def homogeneous_regions(self, img,  intensity = None, saturation = None):
        # gradient calculation with sobel filters
        
        g1 = signal.convolve2d(img, self.sobel1, mode = 'same')
        g2 = signal.convolve2d(img, self.sobel2, mode = 'same')
        g = np.absolute(g1) + np.absolute(g2)
        #cutoff (don't use them for calculation) extreme areas
        if ((intensity is not None)
            and (np.count_nonzero(intensity > 250) < 0.5*img.shape[0]*img.shape[1])
            and (np.count_nonzero(intensity < 5) < 0.5*img.shape[0]*img.shape[1])):
            #dont use positions with high and low intensity I>0.05*255 and I< 0.95*255*3
            g[intensity > 250] = g.max()
            g[intensity <5] = g.max()
        if saturation is not None:
            #dont use positions with high and low sturation S>0.05 and S< 0.95
            g[saturation > 0.95] = g.max()
            g[saturation < 0.05] = g.max()
        #exclude_borders:
        #g[0:int(g.shape[0]*0.15), :] = g.max()
        #g[:,  0:int(g.shape[1]*0.15)] = g.max()
        #g[int(g.shape[0]*0.85):g.shape[0], :] = g.max()
        #g[:, int(g.shape[1]*0.85):g.shape[1]] = g.max()
        threshhold_g = g.min() # threshhold = the G value when the accumulated histogram reaches p% of the whole image
        if (not (g.max() == g.min())):
            # compute the histogram of G # TODO check if could be replaced by sort
            hist_g = np.histogram(g, bins = int(g.max()-g.min()), range = (g.min(),g.max()))
            # calculate threshhold
            p_pixels =  self.p * img.shape[0] * img.shape[1]
            sum_pixels = 0
            for gi, g_value in enumerate(hist_g[0]):
                sum_pixels += g_value
                if sum_pixels > p_pixels:
                    threshhold_g = g.min()+gi
                    break
        masked_img = np.ma.masked_where(g > threshhold_g, img)
        return masked_img


    def estimateNoise(self,img):
        #see Immerkaer Fast Noise Estimation Algorithm
        H, W = img.shape
        conv = signal.convolve2d(img, self.laplacian, mode='valid')
        abs_values = np.absolute(conv)
        sigma = np.sum(np.sum(abs_values))
        return sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    def estimateNoiseExtend(self, img, intensity=None, saturation=None):
        img_suppressed_structures = signal.convolve2d(img, self.laplacian, mode='valid')
       
        intensity = None if intensity is None else intensity[1:-1,1:-1] #pixels lost by convolution
        saturation = None if saturation is None else saturation[1:-1,1:-1] #pixels lost by convolution
        edge_mask = self.homogeneous_regions(img[1:-1,1:-1], intensity, saturation).mask

        abs_residuals = np.ma.array(np.absolute(img_suppressed_structures), mask = edge_mask)
        # print("residuals")
        # plt.imshow(abs_residuals)
        # plt.show()

        # print('edge_mask')
        # plt.imshow(edge_mask)
        # plt.show()
        summed_abs_residuals = np.sum(np.ma.sum(abs_residuals))
        N = (abs_residuals.shape[0]*abs_residuals.shape[1]) - np.count_nonzero(edge_mask)
        #print(summed_abs_residuals, np.count_nonzero(edge_mask), N)
        #print(summed_abs_residuals * np.sqrt(0.5 * np.pi) / (6 * N))
        return summed_abs_residuals * np.sqrt(0.5 * np.pi) / (6 * N)

    def __call__(self,x, intensity=None, saturation=None):
        return self.estimateNoiseExtend(x,intensity, saturation)