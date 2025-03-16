
import torch
import torch.nn as nn
from device import device

def ocdvae_loss(out, labels):
    classification_samples, output_samples, z_mean, z_std = out
    target_classes, target_reconstruction = labels
    return unified_loss_function(classification_samples, target_classes, output_samples, target_reconstruction, z_mean, z_std, device)

def vae_loss(out,labels):
    _, output_samples, z_mean, z_std = out
    target_classes, target_reconstruction = labels
    #print(output_samples.shape,target_reconstruction.shape)

    recon_loss = nn.BCELoss(reduction='sum')
    BCE = recon_loss(output_samples, target_reconstruction)#/ torch.numel(target_reconstruction)

    KLD = -0.5 * torch.sum(1 + z_std - z_mean.pow(2) - z_std.exp())#/ torch.numel(z_mean)
    return BCE+KLD
    #Place-holders for the final loss values over all latent space samples
    #recon_losses = torch.zeros(output_samples.size(0)).to(device)
    """
    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the corresponding loss. Normalize the losses.
    #for i in range(output_samples.size(0)):
    #    recon_losses[i] = recon_loss(output_samples[i], target_reconstruction) #/ torch.numel(target_reconstruction)

    # average the loss over all samples per input
    #rl = torch.mean(recon_losses, dim=0)
    rl = recon_loss(output_samples[0], target_reconstruction)/ torch.numel(target_reconstruction)
    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + z_std ** 2) - (z_mean ** 2) - (z_std ** 2)) / torch.numel(z_mean)

    var_beta = 0.1
    loss = rl + var_beta * kld
    return loss  # cl, rl, kld
    """

'''
This function is adapted from https://github.com/MrtnMndt/OpenVAE_ContinualLearning/blob/31fb1a59c2db6af1dbe1df474a75cd8b5127dd6d/lib/Training/loss_functions.py
Licensed under the MIT License (https://github.com/MrtnMndt/OpenVAE_ContinualLearning/blob/31fb1a59c2db6af1dbe1df474a75cd8b5127dd6d/LICENSE)
Copyright (c) 2019 Martin Mundt
'''
def unified_loss_function(output_samples_classification, target, output_samples_recon, inp, mu, std, device):
    """
    Computes the unified model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of variational samples
    is one per default, as specified in the command line parser and typically is how VAE models and also our unified
    model is trained. We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    #if args.autoregression:
    #    recon_loss = nn.CrossEntropyLoss(reduction='sum')
    #else:
    recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon, inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)
    
    var_beta= 0.1
    loss = cl + rl  + var_beta * kld
    return loss# cl, rl, kld
