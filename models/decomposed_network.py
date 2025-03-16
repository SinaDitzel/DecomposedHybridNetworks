import torch
import torch.nn as nn
import torch.nn.functional as F
from .nconv import NConv2d, ConfMaxPool, NSequential
import math
import collections
from collections import OrderedDict

from device import device


class Network_Config():
    def __init__(self, encoder, encoder_output_dim, decoder=None):
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        self.decoder = decoder
        # TODO classifier


# CONFIG_NConvNetEnc = Network_Config(encoder=[("n",64,(5,5),(1,1)), ("p",(2,2)), ("n",128,(3,3),(1,1)), ("p",(2,2))], #NConc layers
#                        [("c",128,3,1), ("p",(2,2))]) #"standart" CNN layers
CONFIG_ConvNetEnc = Network_Config(encoder=[("c", 64, (5, 5), (1, 1), (0, 0)),
                                            ("p", (2, 2)),
                                            ("c", 128, (3, 3), (1, 1), (0, 0)),
                                            ("p", (2, 2)),
                                            ("c", 128, (3, 3), (1, 1), (0, 0)),
                                            ("p", (2, 2))],
                                   encoder_output_dim=lambda input_size: (128, ((((((input_size-4)//2)-2)//2)-2)//2) , ((((((input_size-4)//2)-2)//2)-2)//2)))

# sCNN architecture inspired by WAE-DCGAN from https://arxiv.org/pdf/1511.06434.pdf but without the GAN component.
CONFIG_DCGAN = Network_Config(encoder=[("c", 128, (3, 3), (2, 2), (1, 1)),
                                       ("c", 256, (3, 3), (2, 2), (1, 1)),
                                       ("c", 512, (3, 3), (2, 2), (1, 1)),
                                       ("c", 512, (3, 3), (2, 2), (1, 1)),
                                       ("c", 1024, (3, 3), (2, 2), (1, 1))],
                              encoder_output_dim=lambda input_size: (1024,2,2) if input_size==48 else (1024, 4, 4),  # (1024,3,3)#TODO
                              decoder=[("tc", 512, (3, 3), (2, 2), (1, 1)),
                                       ("tc", 512, (3, 3), (2, 2), (1, 1)),
                                       ("tc", 256, (3, 3), (2, 2), (1, 1)),
                                       ("tc", 128, (3, 3), (2, 2), (1, 1)),
                                       ("tc", 3, (3, 3), (2, 2), (1, 1))])

CONFIG_SMALL_DCGAN = Network_Config(encoder=[("c", 64, (3, 3), (2, 2), (1, 1)),
                                             ("c", 128, (3, 3), (2, 2), (1, 1)),
                                             ("c", 128, (3, 3), (2, 2), (1, 1)),
                                             ("c", 256, (3, 3), (2, 2), (1, 1))],
                                    encoder_output_dim=(256, 3, 3),
                                    decoder=[("tc", 256, (3, 3), (2, 2), (1, 1)),
                                             ("tc", 128, (3, 3), (2, 2), (1, 1)),
                                             ("tc", 64, (3, 3), (2, 2), (1, 1)),
                                             ("tc", 3, (3, 3), (2, 2), (1, 1))])


class NCNN(nn.Module):
    def __init__(self, config, in_c=3, nconv_layers=0, norm_eps=1e-4, vae=False):
        super(NCNN, self).__init__()
        self.in_c = in_c
        cnn_list = []
        ncnn_list = []
        nc = 0
        for c in config:
            type = c[0]
            if type == "c":  # convolution
                _, out_c, k_size, stride, pad = c
                if nc < nconv_layers:  # ncnn layers
                    nconv = NConv2d(in_c, out_c, k_size, stride, padding=pad)  # out: 32x32
                    nc += 1
                    ncnn_list.append(nconv)
                else:  # cnn layers
                    if vae:
                        conv = nn.Sequential(nn.Conv2d(in_c, out_c, k_size, stride=stride, padding=pad, bias=False),
                                             # out 256,7,7
                                             nn.ReLU(inplace=True))
                    else:
                        conv = nn.Sequential(nn.Conv2d(in_c, out_c, k_size, stride=stride, padding=pad, bias=False),
                                             # out 256,7,7
                                             nn.BatchNorm2d(num_features=out_c, eps=norm_eps),
                                             nn.ReLU(inplace=True))
                    cnn_list.append(conv)
                in_c = out_c
            elif type == "tc":  # transposed convolution
                _, out_c, k_size, stride, padding = c
                output_padding = (0 if k_size[0] % 2 == 0 else 1)
                if vae:
                    tconv = nn.Sequential(nn.ReLU(True),
                                          nn.ConvTranspose2d(in_c, out_c, k_size, stride=stride, padding=padding,
                                                             output_padding=output_padding, bias=False)
                                          )
                else:
                    tconv = nn.Sequential(nn.BatchNorm2d(in_c),
                                          nn.ReLU(True),
                                          nn.ConvTranspose2d(in_c, out_c, k_size, stride=stride, padding=padding,
                                                             output_padding=output_padding, bias=False))
                cnn_list.append(tconv)
                in_c = out_c
            elif type == "p":  # pool
                _, size = c
                if len(cnn_list) == 0 and len(ncnn_list) > 0:  # conf pooling
                    ncnn_list.append(ConfMaxPool(size))
                else:
                    cnn_list.append(nn.MaxPool2d(size))
        self.ncnn = NSequential(*ncnn_list)
        self.cnn = nn.Sequential(*cnn_list)

    def forward(self, x):
        if len(self.ncnn) > 0:
            c = x[:, self.in_c:]
            x = x[:, :self.in_c]
            x, c = self.ncnn(x, c)
        x = self.cnn(x)
        return x


kernel_size = 4  # (4, 4) kernel
init_channels = 32  # initial number of filters
latent_dim = 16  # latent dimension for sampling

class ConvVAE(nn.Module):
    def __init__(self, image_channels=3):
        super(ConvVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2, out_channels=init_channels * 4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=init_channels * 8, kernel_size=kernel_size,
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.encoder_output_dim = 64
        self.fc1 = nn.Linear(init_channels * 8, self.encoder_output_dim)
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1024)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels * 8, kernel_size=kernel_size,
            stride=2, padding=1  # TODO make sure these changes are okay
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8, out_channels=init_channels * 4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4, out_channels=init_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        # encoding
        #print('input shape',x.shape)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        #print('Encoded',x.shape)

        hidden = self.fc1(x)
        #print('hidden',hidden.shape)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # print('mu',mu.shape,log_var.shape)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print('z',z.shape)
        z = self.fc2(z)
        # print('z2',z.shape)

        z = z.reshape(-1, 64, 4, 4)

        # decoding
        x = F.relu(self.dec1(z))

        # print('x1',x.shape)
        x = F.relu(self.dec2(x))
        #print('x2',x.shape)

        x = F.relu(self.dec3(x))
        #print('x3',x.shape)

        x = F.relu(self.dec4(x))
        #print('x4',x.shape)

        reconstruction = torch.sigmoid(self.dec5(x))
        #print('rec',reconstruction.shape)
        return z, reconstruction, mu, log_var


class DecomposedNetwork(nn.Module):

    def __init__(self, num_classes, decompositions, batch_size, input_size, ocdvae=False, vae=False, nconv_layers=2 ):
        super(DecomposedNetwork, self).__init__()
        if ocdvae or vae:
            net_config = CONFIG_DCGAN
        else:
            net_config = CONFIG_ConvNetEnc
        self.decompositions = decompositions
        self.batch_norm = 1e-4
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.latent_dim = 128

        self.vae = vae
        self.ocdevae = ocdvae

        # encoder
        self.encoder = OrderedDict()
        self.encode_latent_mu = OrderedDict()
        self.encode_latent_std = OrderedDict()
        self.encoding_shape = net_config.encoder_output_dim(input_size)
        print('input_size', input_size, net_config.encoder_output_dim(input_size))
        self.num_decompositions = max(1, len(decompositions))
        for i in range(self.num_decompositions):
            if len(self.decompositions) > 0:
                conf, channels = self.decompositions[i].conf, self.decompositions[i].channels
                name = str(self.decompositions[i])
            else:
                conf, channels = False, 3
                name = ""
            nconv_layers_decomp = nconv_layers if conf else 0
            self.encoder.update({"enc_%s" % (name): NCNN(net_config.encoder, channels, nconv_layers_decomp, vae= self.vae)})
            # latent
            if self.ocdevae or self.vae:
                self.encode_latent_mu.update({"enc_mu_%s" % (name): nn.Linear(math.prod(self.encoding_shape),
                                                                              self.latent_dim, bias=False)})
                self.encode_latent_std.update({"enc_std_%s" % (name): nn.Linear(math.prod(self.encoding_shape),
                                                                                self.latent_dim, bias=False)})
        # add all encoders to model (so that weights ar updatet,...)
        self.encoder = nn.Sequential(self.encoder)
        if self.ocdevae or self.vae:
            self.encode_latent_mu = nn.Sequential(self.encode_latent_mu)
            self.encode_latent_std = nn.Sequential(self.encode_latent_std)
            self.decode_latent = nn.Linear(self.latent_dim * self.num_decompositions, math.prod(self.encoding_shape))

            # decoder
            self.decoder = NCNN(net_config.decoder, self.encoding_shape[0],vae= self.vae)

        # classifier
        if not self.vae:
            if self.ocdevae:
                classifier_input_dim = self.latent_dim * self.num_decompositions
            else:
                classifier_input_dim = math.prod(self.encoding_shape) * self.num_decompositions
            self.classifier = nn.Sequential(nn.Dropout(0.25),
                                            nn.Linear(classifier_input_dim, 128, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.25),
                                            nn.Linear(128, num_classes, bias=True))
        #else:
        #    self.vae_model = ConvVAE(image_channels=3)

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        if self.ocdevae or self.vae:
            return self.forward_vae(x)
        #elif self.vae:
        #    return self.vae_model(x)
        # elif self.vae:
        #    return self.vae_model(x)
        # encoder
        feature_vector = []
        last_c = 0
        for i in range(self.num_decompositions):
            if len(self.decompositions) > 0:
                conf, channels = self.decompositions[i].conf, self.decompositions[i].channels
            else:
                conf, channels = False, 3
            tmp_c = last_c + channels * (2 if conf else 1)
            x_decompi = self.encoder[i](x[:, last_c:tmp_c])
            x_decompi = x_decompi.reshape(x_decompi.size(0), -1)
            feature_vector.append(x_decompi)
            last_c = tmp_c
        feature_vector = torch.cat(feature_vector, dim=1)
        # classifier
        c = self.classifier(feature_vector)
        return c

    def forward_vae(self, x):
        # encoder
        mus, stds = [], []
        last_c = 0
        for i in range(self.num_decompositions):
            if len(self.decompositions) > 0:
                conf, channels = self.decompositions[i].conf, self.decompositions[i].channels
            else:
                conf, channels = False, 3
            tmp_c = last_c + channels * (2 if conf else 1)
            x_decompi = self.encoder[i](x[:, last_c:tmp_c])
            x_decompi = x_decompi.reshape(x_decompi.size(0), -1)
            mus.append(self.encode_latent_mu[i](x_decompi))
            stds.append(self.encode_latent_std[i](x_decompi))
            last_c = tmp_c
        mu = torch.cat(mus, dim=1)
        std = torch.cat(stds, dim=1)

        # reparameterization trick
        eps = torch.normal(torch.zeros(self.latent_dim * self.num_decompositions), std=1.0).to(device)
        z = mu.add(eps.mul(std))

        # decoder
        out = self.decode_latent(z)
        out = out.reshape(z.size(0), self.encoding_shape[0], self.encoding_shape[1], self.encoding_shape[2])
        out = self.decoder(out)#.unsqueeze(dim=0)
        out = torch.sigmoid(out)

        if self.ocdevae:
            # classifier
            c = self.classifier(z).unsqueeze(dim=0)

            # output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
            #                             self.patch_size).to(self.device)
            # classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
            # for i in range(self.num_samples):
            # output_samples[i] = self.decode(z)
            # classification_samples[i] = self.classifier(z)
            return c, out, mu, std
        else:
            return z, out, mu, std
