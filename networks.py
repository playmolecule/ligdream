# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


########################
# AutoEncoder Networks #
########################

class LigandVAE(nn.Module):
    def __init__(self, nc=5, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        super(LigandVAE, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv3d(nc, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(32)

        self.e2 = nn.Conv3d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm3d(32)

        self.e3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(64)

        self.e4 = nn.Conv3d(64, ndf * 4, 3, 2, 1)
        self.bn4 = nn.BatchNorm3d(ndf * 4)

        self.e5 = nn.Conv3d(ndf * 4, ndf * 4, 3, 2, 1)
        self.bn5 = nn.BatchNorm3d(ndf * 4)

        self.fc1 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)
        self.fc2 = nn.Linear(512 * 3 * 3 * 3, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, 512 * 3 * 3 * 3)

        # up5
        self.d2 = nn.ConvTranspose3d(ndf * 4, ndf * 4, 3, 2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm3d(ndf * 4, 1.e-3)

        # up 4
        self.d3 = nn.ConvTranspose3d(ndf * 4, ndf * 2, 3, 2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm3d(ndf * 2, 1.e-3)

        # up3 12 -> 12
        self.d4 = nn.Conv3d(ndf * 2, ndf, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm3d(ndf, 1.e-3)

        # up2 12 -> 24
        self.d5 = nn.ConvTranspose3d(ndf + 32, 32, 3, 2, padding=1, output_padding=1)
        self.bn9 = nn.BatchNorm3d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv3d(64, nc, 3, 1, padding=1)

        # Condtional encoding
        self.ce1 = nn.Conv3d(3, 32, 3, 1, 1)
        self.ce2 = nn.Conv3d(32, 32, 3, 2, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, 512 * 3 * 3 * 3)
        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar, factor):
        std = logvar.mul(0.5).exp_()
        if self.use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return (eps.mul(std) * factor).add_(mu)

    def decode(self, z, cond_x):
        # Conditional block
        cc1 = self.relu(self.ce1(cond_x))
        cc2 = self.relu(self.ce2(cc1))

        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ndf * 4, 3, 3, 3)
        h2 = self.leakyrelu(self.bn6(self.d2((h1))))
        h3 = self.leakyrelu(self.bn7(self.d3(h2)))
        h4 = self.leakyrelu(self.bn8(self.d4(h3)))
        h4 = torch.cat([h4, cc2], dim=1)
        h5 = self.leakyrelu(self.bn9(self.d5(h4)))
        h5 = torch.cat([h5, cc1], dim=1)
        return self.sigmoid(self.d6(h5))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view())
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x, cond_x, factor=1.):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar, factor=factor)
        res = self.decode(z, cond_x)
        return res, mu, logvar


#######################
# Captioning Networks #
#######################

class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids
