import numpy as np 
from scipy import  stats

class Confidence():
    def __init__(self, lambdas=lambda d: [0,1,10,np.median(d)*0.5], weight=[0.25,0.25,0.25,0.25], k =2, prior_H0=0.5):
        self.lambdas = lambdas
        self.weights = weight
        self.k = 2 # degrees of freeedom of random variable confidence is calculated for
        self.prior_H0 = prior_H0

    def moments_noncentral_chi(self, min_lambda, max_lambda, k, debug =False):
        a = min_lambda
        b = max_lambda
        if b == 0:
            mu = 0
        else:
            mu = k+((a+b)/2)
        var = 2*k+2*(a+b)+1/12*(a*a+b*b+2*a*b)
        return mu, var

    def posterior_H1_given_d(self, mah_dist, max_lambdaH0, min_lambda_H1, max_lambdaH1):
        min_lambdaH0 = 0
     
        mu_d_H0, var_d_H0 = self.moments_noncentral_chi(min_lambdaH0, max_lambdaH0, self.k)
        mu_d_H1, var_d_H1 = self.moments_noncentral_chi(min_lambda_H1, max_lambdaH1, self.k)

        likelihood_H0 = stats.norm.pdf(mah_dist, mu_d_H0, np.sqrt(var_d_H0))
        likelihood_H1 = stats.norm.pdf(mah_dist, mu_d_H1, np.sqrt(var_d_H1))
        out = np.full(likelihood_H1.shape,0.5)
        out[mah_dist>mu_d_H0] = 1
        marginal = likelihood_H1 * (1-self.prior_H0) + likelihood_H0 * self.prior_H0
        p = np.divide((likelihood_H1 * (1-self.prior_H0)), marginal, out=out, where=(marginal>1e-9))
        return p

    def __call__(self, d):
        conf = np.expand_dims(self.posterior_H1_given_d(d, self.lambdas(d)[0], self.lambdas(d)[0], np.median(d)*1.75),axis=2) * self.weights[0]
        for l,w in zip(self.lambdas(d)[1:], self.weights[1:]):
            tmp_conf = self.posterior_H1_given_d(d, l, l, np.median(d)*1.75)*w
            conf = np.concatenate([conf, np.expand_dims(tmp_conf, axis=2)], axis=2)
        conf = np.sum(conf, axis=2, keepdims=True)
        return conf