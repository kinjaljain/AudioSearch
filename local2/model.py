import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.contiguous()
        x = x.view(batch_size * time_steps, -1)
        x = self.module(x)
        x = x.contiguous()
        x = x.view(batch_size, time_steps, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class Encoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        prev_channels = 80
        total_scale = 1
        pad_left = 0
        self.skips = []
        for stride, ksz, dilation_factor in layer_specs:
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)

            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            self.skips.append(skip)
            total_scale *= stride
        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)

    def forward(self, samples):
        x = samples.transpose(1, 2)
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec
            x1 = conv_wide(x)
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = conv_1x1(x2)
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip + x3.size(2) * stride].view(
                    x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        return x.transpose(1, 2)

class VectorQuant(nn.Module):
    """
        Input: (N, samples, n_channels, vec_len) numeric tensor
        Output: (N, samples, n_channels, vec_len) numeric tensor
    """

    def __init__(self, n_channels, n_classes, vec_len, normalize=False):
        super().__init__()
        if normalize:
            target_scale = 0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(
            torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        self.n_classes = n_classes
        self.after_update()

    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True:
            # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
        else:
            entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        # logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(
        # dim=0, index=index1).std()}')
        return out0, out1, out2, entropy

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))

    def get_latents(self, x0):
        b = x0.shape[0]
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
            # logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True:  # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            # logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        return index1.view(b, -1)

    def get_quantizedindices(self, x0):
        x = x0
        embedding = self.embedding0
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        entropy = 0
        print("Predicted quantized Indices are: ", (index.squeeze(1) + self.offset).cpu().numpy())
        print('\n')

class AudioFinder(nn.Module):
    def __init__(self, normalize_vq=False, noise_x=False, noise_y=False):
        super(AudioFinder, self).__init__()
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (1, 4, 1),
        ]
        self.encoder = Encoder(80, encoder_layers)
        self.vq = VectorQuant(1, 512, 80, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        # self.searchNquery2label = SequenceWise(nn.Linear(80, 1))
        # in case of CE loss u need output (N,C)
        self.searchNquery2label = SequenceWise(nn.Linear(80, 2))



    # def forward(self, search, pos_query, neg_queries):
        # encoded_search = self.encoder(search)
        # encoded_pos_query = self.encoder(pos_query)
        # encoded_neg_queries = []
        # for neg_query in neg_queries:
        #     # print(neg_query.shape)
        #     encoded_neg_queries.append(self.encoder(neg_query))
        #
        # # print("Shape of search, encoded_pos_query, encoded_neg_query: ",
        # #       encoded_search.shape, encoded_pos_query.shape, encoded_neg_queries[0].shape)
        #
        # discrete, vq_pen, encoder_pen, entropy = self.vq(encoded_search.unsqueeze(2))
        #
        # discrete = discrete.squeeze(2)
        #
        # # check torch.interpolate instead of this if very slow
        # discrete_1 = discrete.shape[1]
        # query_1 = encoded_pos_query.shape[1]
        # scale_factor = discrete_1 // query_1
        # encoded_pos_query = encoded_pos_query.repeat([1, scale_factor, 1])
        # for i, encoded_neg_query in enumerate(encoded_neg_queries):
        #     encoded_neg_queries[i] = encoded_neg_query.repeat([1, scale_factor, 1])
        # pad_val = discrete_1 - encoded_pos_query.shape[1]
        # p2d = [0, 0, 0, pad_val]  # pad last dim by (0, 0) and 2nd to last by (0, pad_val)
        # encoded_pos_query = F.pad(encoded_pos_query, p2d, "constant", 0)
        # for i, encoded_neg_query in enumerate(encoded_neg_queries):
        #     pad_val = discrete_1 - encoded_neg_query.shape[1]
        #     p2d = [0, 0, 0, pad_val]
        #     encoded_neg_queries[i] = F.pad(encoded_neg_query, p2d, "constant", 0)
        #
        # # print("Shape of discrete, encoded_pos_query, encoded_neg_query: ",
        # #       discrete.shape, encoded_pos_query.shape, encoded_neg_queries[0].shape)
        # prediction_pos = torch.tanh(self.searchNquery2label(encoded_pos_query + discrete))
        # prediction_negs = []
        # for i, encoded_neg_query in enumerate(encoded_neg_queries):
        #     prediction_neg = torch.tanh(self.searchNquery2label(encoded_neg_query + discrete))
        #     prediction_neg, _ = torch.max(prediction_neg, dim=1)
        #     # print("prediction_neg_shape:",  prediction_neg.shape)
        #     prediction_negs.append(prediction_neg.squeeze(1))
        # prediction_pos, _ = torch.max(prediction_pos, dim=1)
        # # print("Shape of predictions: ", prediction_pos.shape, prediction_negs[0].shape)
        # return prediction_pos.squeeze(1), prediction_negs
        #

    def forward(self, search, query):
        encoded_search = self.encoder(search)
        encoded_query = self.encoder(query)
        discrete, vq_pen, encoder_pen, entropy = self.vq(encoded_search.unsqueeze(2))
        discrete = discrete.squeeze(2)
        discrete_1 = discrete.shape[1]
        query_1 = encoded_query.shape[1]
        scale_factor = discrete_1 // query_1
        encoded_query = encoded_query.repeat([1, scale_factor, 1])
        pad_val = discrete_1 - encoded_query.shape[1]
        p2d = [0, 0, 0, pad_val]  # pad last dim by (0, 0) and 2nd to last by (0, pad_val)
        encoded_query = F.pad(encoded_query, p2d, "constant", 0)
        prediction = torch.tanh(self.searchNquery2label(encoded_query + discrete))
        prediction, _ = torch.max(prediction, dim=1)
        print("Shape of predictions: ", prediction.shape)
        return prediction.squeeze(1)
