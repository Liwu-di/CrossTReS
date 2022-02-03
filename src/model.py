import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import logging
import datetime
import copy
from dgl import function as fn
import dgl
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from collections import OrderedDict
# TODO: replace the VF.lstm in this file 
# try implement LSTM and see which is better
from torch._VF import lstm
# ordered dict remembers the sequence where keys are inserted

# this is a hand-implemented functional LSTM
def functional_lstm(input_, w_ih, w_hh, b_ih, b_hh, hidden_size_):
    # This function implements a functional LSTM forwarder. 
    time_steps = input_.shape[0]
    bsize = input_.shape[1]
    # initialize hidden 
    h = torch.zeros(1, bsize, hidden_size_, device = w_ih.device)
    c = torch.zeros(1, bsize, hidden_size_, device = w_ih.device)
    outputs = []
    states = []
    for i in range(time_steps):
        from_i = F.linear(input_[i].squeeze(), w_ih, b_ih)
        from_h = F.linear(h.squeeze(), w_hh, b_hh)
        chunks = from_i + from_h
        it, ft, gt, ot = chunks.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        c = ft * c + it * gt
        h = ot * torch.tanh(c)
        # print(h.shape)
        outputs.append(h)
    return torch.cat(outputs, 0), (h, c)


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x
    
    def functional_forward(self, x, weights = None, bn_vars = None, bn_training = True):
        # print('weights', weights.keys())
        # print('bn_vars', bn_vars.keys())
        if weights is None:
            # weights = OrderedDict(self.named_parameters())
            weights = OrderedDict(self.named_parameters())
            # state_dict includes bn.running_mean, bn.running_var, bn.num_batches_tracked
            # while named_parameters() does not
            bn_vars = OrderedDict()
            for k in self.state_dict():
                if 'running_mean' in k or 'running_var' in k:
                    bn_vars[k] = self.state_dict()[k]
                
        # keys: 
        # bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, bn1.num_batches_tracked
        # conv1.weight, conv1.bias 
        # bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var, bn2.num_batches_tracked 
        # conv2.weight, conv2.bias 
        z = F.batch_norm(x, bn_vars['bn1.running_mean'], bn_vars['bn1.running_var'], weights['bn1.weight'], weights['bn1.bias'], training = bn_training)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv1.weight'], weights['conv1.bias'], stride = 1, padding = 1)
        z = F.batch_norm(z, bn_vars['bn2.running_mean'], bn_vars['bn2.running_var'], weights['bn2.weight'], weights['bn2.bias'], training = bn_training)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv2.weight'], weights['conv2.bias'], stride = 1, padding = 1)
        return z+x 

class ResUnit_nobn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit_nobn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        z = F.relu(x)
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x
    
    def functional_forward(self, x, weights = None, bn_vars = None, bn_training = None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
            # weights = OrdererDict(self.state_dict())
            # state_dict includes bn.running_mean, bn.running_var, bn.num_batches_tracked
            # while named_parameters() does not
        # keys: 
        # conv1.weight, conv1.bias 
        # conv2.weight, conv2.bias 
        z = F.relu(x)
        z = F.conv2d(z, weights['conv1.weight'], weights['conv1.bias'], stride = 1, padding = 1)
        z = F.relu(z)
        z = F.conv2d(z, weights['conv2.weight'], weights['conv2.bias'], stride = 1, padding = 1)
        return z+x 

class BatchGATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(BatchGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, verbose = False):
        """Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        num_node = feat.shape[0]
        num_batch = feat.shape[1]
        
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(num_node, num_batch, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(num_node, num_batch, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                num_node, num_batch, self._num_heads, self._out_feats)
            
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # el: (1, num_batch, num_heads)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        if verbose:
            print(graph.edata['a'])
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(num_node, num_batch, -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst



class STResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(STResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.layers = []
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.layers.append(ResUnit(in_channels = 64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, X, spatial_mask = None):
        X = self.conv1(X)
        for layer in self.layers:
            X = layer(X)
        X = self.conv2(X)
        return torch.sigmoid(X)
    
    def functional_forward(self, X, spatial_mask = None, weights = None, bn_vars = None, bn_training = True):
        if weights is None:
            # weights = OrderedDict(self.named_parameters())
            weights = OrderedDict(self.named_parameters())
            # state_dict includes bn.running_mean, bn.running_var, bn.num_batches_tracked
            # while named_parameters() does not
            bn_vars = OrderedDict()
            for k in self.state_dict():
                if k not in weights and "num_batches_tracked" not in k:
                    bn_vars[k] = self.state_dict()[k]
        X = F.conv2d(X, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
        for i in range(self.num_blocks):
            # forward residual blocks 
            sub_weights = OrderedDict()
            for k in weights.keys():
                if k.startswith("layers.%d" % i):
                    stripped_key = k.lstrip("layers.%d." % i)
                    sub_weights[stripped_key] = weights[k]
            sub_bnvars = OrderedDict()
            for k in bn_vars.keys():
                if k.startswith("layers.%d" % i):
                    stripped_key = k.lstrip("layers.%d." % i)
                    sub_bnvars[stripped_key] = bn_vars[k]
            X = self.layers[i].functional_forward(X, sub_weights, sub_bnvars, bn_training)
        X = F.conv2d(X, weights['conv2.weight'], weights['conv2.bias'], stride=1, padding=1)
        return torch.sigmoid(X)

class STResNet_nobn(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, sigmoid_act = False):
        super(STResNet_nobn, self).__init__()
        self.sigmoid_act = sigmoid_act
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.layers = []
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.layers.append(ResUnit_nobn(in_channels = 64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X, spatial_mask = None):
        X = self.conv1(X)
        for layer in self.layers:
            X = layer(X)
        X = self.conv2(X)
        if self.sigmoid_act:
            return torch.sigmoid(X)
        else:
            return X

    def functional_forward(self, X, spatial_mask = None, weights = None, bn_vars = None, bn_training = None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
            # weights = OrdererDict(self.state_dict())
            # state_dict includes bn.running_mean, bn.running_var, bn.num_batches_tracked
            # while named_parameters() does not
        X = F.conv2d(X, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
        for i in range(self.num_blocks):
            # forward residual blocks 
            sub_weights = OrderedDict()
            for k in weights.keys():
                if k.startswith("layers.%d" % i):
                    stripped_key = k.lstrip("layers.%d." % i)
                    sub_weights[stripped_key] = weights[k]
            X = self.layers[i].functional_forward(X, sub_weights)
        X = F.conv2d(X, weights['conv2.weight'], weights['conv2.bias'], stride=1, padding=1)
        if self.sigmoid_act:
            return torch.sigmoid(X)
        else:
            return X
    
class STNet(nn.Module):
    def __init__(self, num_channels, num_convs, spatial_mask):
        super(STNet, self).__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        self.num_convs = num_convs
        for i in range(num_convs):
            self.layers.append(ResUnit(in_channels=64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
        self.linear = nn.Linear(128 * 2, 1)
    
    def forward(self, X, spatial_mask = None, return_feat = False):
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        # split according to lag
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i*self.num_channels:(i+1)*self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()# [:, spatial_mask.view(-1), :].contiguous()
            outs.append(z.view(-1, 64))
        # outs: # lag * (B, 64, lng, lat)
        # lstm requires (seq_len, batch_size, 64)
        z = torch.stack(outs, dim = 0)
        # print('z', z.shape)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :] 
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1,128)], dim = -1)
        # print("temporal", temporal.shape)
        # batch, # grids, # feat
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        # print("temporal", temporal.shape)
        output = torch.sigmoid(self.linear(temporal_valid)).permute(0, 2, 1)
        # Batch, 1, # validpoints
        if return_feat: 
            return temporal, output
        else:
            return output
    
    def functional_forward(self, X, spatial_mask = None, weights = None, bn_vars = None, bn_training = None, return_feat = False):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
            bn_vars = OrderedDict()
            for k in self.state_dict():
                if k not in weights and "num_batches_tracked" not in k:
                    bn_vars[k] = self.state_dict()[k]
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i*self.num_channels:(i+1)*self.num_channels, :, :]
            z = F.conv2d(input, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
            for j in range(self.num_convs):
                sub_weights = OrderedDict()
                for k in weights.keys():
                    if k.startswith("layers.%d" % j):
                        stripped_key = k.lstrip("layers.%d." % j)
                        sub_weights[stripped_key] = weights[k]
                sub_bnvars = OrderedDict()
                for k in bn_vars.keys():
                    if k.startswith("layers.%d" % j):
                        stripped_key = k.lstrip("layers.%d." % j)
                        sub_bnvars[stripped_key] = bn_vars[k]
                z = self.layers[j].functional_forward(z, sub_weights, sub_bnvars, bn_training)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)
        # The following lines of code uses _VF.lstm
        # max_batch_size = z.shape[1] 
        # h_zeros = torch.zeros(1,
        #                       max_batch_size, 128,
        #                       dtype=z.dtype, device=z.device)
        # c_zeros = torch.zeros(1,
        #                       max_batch_size, 128,
        #                       dtype=z.dtype, device=z.device)
        # hx = (h_zeros, c_zeros)
        flat_weights = [weights['lstm.weight_ih_l0'], weights['lstm.weight_hh_l0'], weights['lstm.bias_ih_l0'], weights['lstm.bias_hh_l0']]
        result = functional_lstm(z, flat_weights[0], flat_weights[1], flat_weights[2], flat_weights[3], 128)
        # with torch.backends.cudnn.flags(enabled=False):
        #     result = lstm(z, hx, flat_weights, True, 1, 0, self.training, False, False)
        temporal_out = result[0]
        temporal_hid, _ = result[1]
        temporal_out = temporal_out[-1:, :] 
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1,128)], dim = -1)
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        output = torch.sigmoid(F.linear(temporal_valid, weights['linear.weight'], weights['linear.bias'])).permute(0, 2, 1)
        if return_feat: 
            return temporal, output
        else:
            return output

class STNet_nobn(nn.Module):
    def __init__(self, num_channels, num_convs, spatial_mask):
        super(STNet_nobn, self).__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        self.bns = []
        self.num_convs = num_convs
        for i in range(num_convs):
            self.layers.append(ResUnit_nobn(64, 64))
        self.layers = nn.ModuleList(self.layers)
        # self.bns = nn.ModuleList(self.layers)
        # previously, the hidden dim of LSTM is 128
        self.lstm = nn.LSTM(64, 128)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 1)
    
    def forward(self, X, spatial_mask = None, return_feat = False):
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        # split according to lag
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i*self.num_channels:(i+1)*self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
                # z = F.relu(z)
                # z = bn(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()# [:, spatial_mask.view(-1), :].contiguous()
            outs.append(z.view(-1, 64))
        # outs: # lag * (B, 64, lng, lat)
        # lstm requires (seq_len, batch_size, 64)
        z = torch.stack(outs, dim = 0)
        # print('z', z.shape)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :] 
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        # print("temporal", temporal.shape)
        # batch, # grids, # feat
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        # print("temporal", temporal.shape)
        hid = F.relu(self.linear1(temporal_valid))
        output = torch.sigmoid(self.linear2(hid)).permute(0, 2, 1)
        # Batch, 1, # validpoints
        if return_feat: 
            return temporal, output
        else:
            return output
    
    def functional_forward(self, X, spatial_mask = None, weights = None, bn_vars = None, bn_training = None, return_feat = False):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        if spatial_mask is None:
            spatial_mask = self.spatial_mask
        num_lag = (X.shape[1] // self.num_channels)
        batch_size = X.shape[0]
        outs = []
        for i in range(num_lag):
            input = X[:, i*self.num_channels:(i+1)*self.num_channels, :, :]
            z = F.conv2d(input, weights['conv1.weight'], weights['conv1.bias'], stride=1, padding=1)
            for j in range(self.num_convs):
                sub_weights = OrderedDict()
                for k in weights.keys():
                    if k.startswith("layers.%d" % j):
                        stripped_key = k.lstrip("layers.%d." % j)
                        sub_weights[stripped_key] = weights[k]
                z = self.layers[j].functional_forward(z, sub_weights)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)
        # forward lstm
        # The following lines of code uses functional lstm
        # max_batch_size = z.shape[1]
        # h_zeros = torch.zeros(1,
        #                       max_batch_size, 128,
        #                       dtype=z.dtype, device=z.device)
        # c_zeros = torch.zeros(1,
        #                       max_batch_size, 128,
        #                       dtype=z.dtype, device=z.device)
        # hx = (h_zeros, c_zeros)
        flat_weights = [weights['lstm.weight_ih_l0'], weights['lstm.weight_hh_l0'], weights['lstm.bias_ih_l0'], weights['lstm.bias_hh_l0']]
        # with torch.backends.cudnn.flags(enabled=False):
        #     result = lstm(z, hx, flat_weights, True, 1, 0, self.training, False, False)
        result = functional_lstm(z, flat_weights[0], flat_weights[1], flat_weights[2], flat_weights[3],128)
        temporal_out = result[0]
        temporal_hid, _ = result[1]
        temporal_out = temporal_out[-1:, :] 
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        temporal_valid = temporal[:, spatial_mask.view(-1), :]
        hid = F.relu(F.linear(temporal_valid, weights['linear1.weight'], weights['linear1.bias']))
        output = torch.sigmoid(F.linear(hid, weights['linear2.weight'], weights['linear2.bias'])).permute(0, 2, 1)
        if return_feat:
            return temporal, output
        else:
            return output
        
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim = 64):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), 
                        nn.ReLU(), 
                        nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        # input z: (4, B, N, F)
        # attention score shape: (4, B, N)
        w = self.project(z) # (4, B, N, 1)
        w = torch.softmax(z, dim = 0)
        return (w * z).sum(0)



class CrossGTPNet(nn.Module):
    # crossgtpnet has three differences compared to STNet
    # A graph encoding source-target relations
    # A GAT module from source to target 
    # Another linear module for prediction
    def __init__(self, num_channels, num_convs, source_embs, target_embs, spatial_mask, topk = 10):
        super().__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        for i in range(num_convs):
            self.layers.append(ResUnit(in_channels = 64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
        self.linear = nn.Linear(128 * 2, 1)
        self.source_linear = nn.Linear(64, 1)
        self.att_module = nn.Sequential(
            nn.Linear(320, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

        # construct graphs
        self.source_embs = source_embs / np.sqrt(1e-5 + (source_embs ** 2).sum(1, keepdims = True))
        self.target_embs = target_embs / np.sqrt(1e-5 + (target_embs ** 2).sum(1, keepdims = True))
        # normalize
        self.cossim = np.matmul(self.source_embs, self.target_embs.T)
        # build knn graph
        edge_list = []
        self.topk = topk
        for j in range(self.cossim.shape[1]):
            cossim = self.cossim[:, j]
            knn = np.argsort(cossim)[-topk:]
            for i in knn:
                edge_list.append((i, j))
                # only from s to t
        print("Cross city similarity graph: %d source, %d target, %d edges" % (self.cossim.shape[0], self.cossim.shape[1], len(edge_list)))
        edge_source = []
        edge_target = []
        for i, j in edge_list:
            edge_source.append(i)
            edge_target.append(j + self.cossim.shape[0])
        # add self loop
        for i in range(self.cossim.shape[0] + self.cossim.shape[1]):
            edge_source.append(i)
            edge_target.append(i)
        self.crosscity_graph = dgl.graph((torch.Tensor(edge_source).long(), torch.Tensor(edge_target).long()))
        self.feat_gatconv = BatchGATConv(in_feats = 256, out_feats = 16, num_heads = 4, residual = True)
        self.gate_gatconv = BatchGATConv(in_feats = 256, out_feats = 1, num_heads = 2, residual = True)
        # self.mmd = MMD_loss()
        self.sematt = SemanticAttention(64)
    
    def forward(self, target_x, source_feats, spatial_mask = None):
        # first forward target_x
        if spatial_mask is None:
            spatial_mask = self.spatial_mask 
        num_lag = (target_x.shape[1] // self.num_channels)
        batch_size = target_x.shape[0] 
        outs = []
        for i in range(num_lag):
            input = target_x[:, i * self.num_channels:(i+1) * self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        target_temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        # target_temporal: (batch_size, # target, 256)
        # run graph attention 
        propagated_feats = []
        for i in range(len(source_feats)):
            source_feat = source_feats[i]
            combined_feat = torch.cat([source_feat, target_temporal], dim = 1).permute(1, 0, 2)
            # compute mmd between source and target feat, to see their difference
            # sampled_source_idx = torch.randint(0, source_feat.shape[1], (32, ))
            # sampled_target_idx = torch.randint(0, target_temporal.shape[1], (32, ))
            # sampled_target_idx2 = torch.randint(0, target_temporal.shape[1], (32, ))
            # print('source_feat', source_feat.shape)
            # print('target_feat', target_temporal.shape)
            # mmd_loss = self.mmd(source_feat[:, sampled_source_idx, :].view(-1, 256), target_temporal[:, sampled_target_idx, :].view(-1, 256))
            # mmd_loss_tt = self.mmd(target_temporal[:, sampled_target_idx2, :].view(-1, 256), target_temporal[:, sampled_target_idx, :].view(-1, 256))
            # print("source target mmd", mmd_loss.item())
            # print("target target mmd", mmd_loss_tt.item())
            propagated_feat = self.feat_gatconv(self.crosscity_graph, combined_feat).flatten(2)
            propagated_feat = propagated_feat.permute(1, 0, 2)[:, self.cossim.shape[0]:, :]
            propagated_feat = F.relu(propagated_feat)[:, spatial_mask.view(-1), :]
            propagated_feats.append(propagated_feat)
        propagated_feats = torch.stack(propagated_feats, dim = 0)
        # print(propagated_feats.shape)
        propagated_feats = self.sematt(propagated_feats)
        # print(propagated_feats.shape)
        # print()
        target_temporal = target_temporal[:, spatial_mask.view(-1), :]
        # print("propagated_feat", propagated_feat.shape)
        # print("target_temporal", target_temporal.shape)
        att_score = self.att_module(torch.cat([target_temporal, propagated_feat], dim = -1))
        att_score = torch.tanh(att_score)
        # print(att_score)
        # if compute_mmd:
        #     return torch.sigmoid(self.linear(target_temporal) + att_score * self.source_linear(propagated_feat)).permute(0, 2, 1), mmd_loss
        # else:
        return torch.sigmoid(self.linear(target_temporal) + att_score * self.source_linear(propagated_feat)).permute(0, 2, 1)
        # return torch.sigmoid(self.linear(target_temporal)).permute(0, 2, 1)

class CrossGTPNet_nobn(nn.Module):
    # crossgtpnet has three differences compared to STNet
    # A graph encoding source-target relations
    # A GAT module from source to target 
    # Another linear module for prediction
    def __init__(self, num_channels, num_convs, source_embs, target_embs, spatial_mask, topk = 10):
        super().__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        for i in range(num_convs):
            self.layers.append(ResUnit_nobn(in_channels = 64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 1)
        self.source_linear = nn.Linear(64, 1)
        self.att_module = nn.Sequential(
            nn.Linear(320, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

        # construct graphs
        self.source_embs = source_embs / np.sqrt(1e-5 + (source_embs ** 2).sum(1, keepdims = True))
        self.target_embs = target_embs / np.sqrt(1e-5 + (target_embs ** 2).sum(1, keepdims = True))
        # normalize
        self.cossim = np.matmul(self.source_embs, self.target_embs.T)
        # build knn graph
        edge_list = []
        self.topk = topk
        for j in range(self.cossim.shape[1]):
            cossim = self.cossim[:, j]
            knn = np.argsort(cossim)[-topk:]
            for i in knn:
                edge_list.append((i, j))
                # only from s to t
        print("Cross city similarity graph: %d source, %d target, %d edges" % (self.cossim.shape[0], self.cossim.shape[1], len(edge_list)))
        edge_source = []
        edge_target = []
        for i, j in edge_list:
            edge_source.append(i)
            edge_target.append(j + self.cossim.shape[0])
        # add self loop
        for i in range(self.cossim.shape[0] + self.cossim.shape[1]):
            edge_source.append(i)
            edge_target.append(i)
        self.crosscity_graph = dgl.graph((torch.Tensor(edge_source).long(), torch.Tensor(edge_target).long()))
        self.feat_gatconv = BatchGATConv(in_feats = 256, out_feats = 16, num_heads = 4, residual = True)
        self.gate_gatconv = BatchGATConv(in_feats = 256, out_feats = 1, num_heads = 2, residual = True)
        # self.mmd = MMD_loss()
        self.sematt = SemanticAttention(64)
    
    def forward(self, target_x, source_feats, spatial_mask = None):
        # first forward target_x
        if spatial_mask is None:
            spatial_mask = self.spatial_mask 
        num_lag = (target_x.shape[1] // self.num_channels)
        batch_size = target_x.shape[0] 
        outs = []
        for i in range(num_lag):
            input = target_x[:, i * self.num_channels:(i+1) * self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        target_temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        # target_temporal: (batch_size, # target, 256)
        # run graph attention 
        propagated_feats = []
        for i in range(len(source_feats)):
            source_feat = source_feats[i]
            combined_feat = torch.cat([source_feat, target_temporal], dim = 1).permute(1, 0, 2)
            # compute mmd between source and target feat, to see their difference
            # sampled_source_idx = torch.randint(0, source_feat.shape[1], (32, ))
            # sampled_target_idx = torch.randint(0, target_temporal.shape[1], (32, ))
            # sampled_target_idx2 = torch.randint(0, target_temporal.shape[1], (32, ))
            # print('source_feat', source_feat.shape)
            # print('target_feat', target_temporal.shape)
            # mmd_loss = self.mmd(source_feat[:, sampled_source_idx, :].view(-1, 256), target_temporal[:, sampled_target_idx, :].view(-1, 256))
            # mmd_loss_tt = self.mmd(target_temporal[:, sampled_target_idx2, :].view(-1, 256), target_temporal[:, sampled_target_idx, :].view(-1, 256))
            # print("source target mmd", mmd_loss.item())
            # print("target target mmd", mmd_loss_tt.item())
            propagated_feat = self.feat_gatconv(self.crosscity_graph, combined_feat).flatten(2)
            propagated_feat = propagated_feat.permute(1, 0, 2)[:, self.cossim.shape[0]:, :]
            propagated_feat = F.relu(propagated_feat)[:, spatial_mask.view(-1), :]
            propagated_feats.append(propagated_feat)
        propagated_feats = torch.stack(propagated_feats, dim = 0)
        # print(propagated_feats.shape)
        propagated_feats = self.sematt(propagated_feats)
        # print(propagated_feats.shape)
        # print()
        target_temporal = target_temporal[:, spatial_mask.view(-1), :]
        # print("propagated_feat", propagated_feat.shape)
        # print("target_temporal", target_temporal.shape)
        att_score = self.att_module(torch.cat([target_temporal, propagated_feat], dim = -1))
        att_score = torch.tanh(att_score)
        # print(att_score)
        # if compute_mmd:
        #     return torch.sigmoid(self.linear(target_temporal) + att_score * self.source_linear(propagated_feat)).permute(0, 2, 1), mmd_loss
        # else:
        return torch.sigmoid(self.linear2(F.relu(self.linear1(target_temporal))) + att_score * self.source_linear(propagated_feat)).permute(0, 2, 1)
        # return torch.sigmoid(self.linear(target_temporal)).permute(0, 2, 1)
        
class BaseNet(nn.Module):
    # BaseNet is essentially an STNet with conv and LSTM, but no prediction
    def __init__(self, num_channels, num_convs):
        super(BaseNet, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        for i in range(num_convs):
            self.layers.append(ResUnit(in_channels=64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
    
    def forward(self, X):
        # we do not mask out invalid regions
        num_lag = X.shape[1] // self.num_channels
        batch_size = X.shape[0] 
        lng = X.shape[2]
        lat = X.shape[3]
        outs = []
        for i in range(num_lag):
            input = X[:, i * self.num_channels:(i+1)*self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)

        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        return temporal.reshape(batch_size, lng, lat, 256).permute(0, 3, 1, 2).contiguous()

class BaseNet_cg(nn.Module):
    # base model for crossgtp consist of two residual blocks
    def __init__(self, num_channels, num_convs):
        super().__init__()
        self.num_channels = num_channels
        self.num_convs = num_convs
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        for i in range(num_convs):
            self.layers.append(ResUnit(in_channels = 64, out_channels = 64))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        num_lag = X.shape[1] 
        batch_size = X.shape[0]
        lng = X.shape[2]
        lat = X.shape[3]
        outs = []
        for i in range(num_lag):
            input = X[:, i * self.num_channels:(i+1)*self.num_channels, :, :]
            z = self.conv1(input)
            for layer in self.layers:
                z = layer(z)
            outs.append(z)
        return outs



class AdaptNet(nn.Module):
    # AdaptNet aims to generate features
    # consist of 2 conv layers
    def __init__(self):
        super(AdaptNet, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
    
    def forward(self, X):
        z1 = self.conv1(X)
        z1 = F.relu(z1)
        z2 = self.conv2(z1)
        z2 = F.relu(z2)
        return [z1, z2]

class AdaptNet_cg(nn.Module):
    # AdaptNet in crossgtp consist of 1 residual block and 1 lstm
    def __init__(self, num_channels, num_convs):
        super().__init__()
        self.num_channels = num_channels
        self.num_convs = num_convs
        self.layers = []
        for i in range(num_convs):
            self.layers.append(ResUnit(in_channels = 64, out_channels = 64)) 
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)

    def forward(self, ins):
        batch_size = ins[0].shape[0]
        # ins is a list of Tensors
        outs = []
        for i in range(len(ins)):
            z = ins[i]
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))
        z = torch.stack(outs, dim = 0)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim = -1)
        return outs, temporal # list of (b, N, 64), (B, N, F)

class PredNet(nn.Module):
    # PredNet aims to generate predictions
    def __init__(self):
        super(PredNet, self).__init__()
        self.conv = nn.Conv2d(64, 1, 1, 1, 0)
    
    def forward(self, X):
        return self.conv(X)

class PredNet_cg(nn.Module):
    # PredNet in crossgtp consist of 1 linear layer
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 1)

    def forward(self, temporal):
        return self.linear(temporal).permute(0, 2, 1)
        # it takes B, N, F in, and outputs B, 1, N

class CrossGTPNet2(nn.Module):
    # CrossGTPNet2 consist of a GAT layer and a prediction layer 
    def __init__(self, in_dim, source_embs, target_embs, spatial_mask, topk = 10):
        super().__init__()
        # construct graphs
        self.in_dim = in_dim
        self.spatial_mask = spatial_mask.bool()
        self.source_embs = source_embs / np.sqrt(1e-5 + (source_embs ** 2).sum(1, keepdims = True))
        self.target_embs = target_embs / np.sqrt(1e-5 + (target_embs ** 2).sum(1, keepdims = True))
        # normalize
        self.cossim = np.matmul(self.source_embs, self.target_embs.T)
        # build knn graph
        edge_list = []
        self.topk = topk
        for j in range(self.cossim.shape[1]):
            cossim = self.cossim[:, j]
            knn = np.argsort(cossim)[-topk:]
            for i in knn:
                edge_list.append((i, j))
                # only from s to t
        print("Cross city similarity graph: %d source, %d target, %d edges" % (self.cossim.shape[0], self.cossim.shape[1], len(edge_list)))
        edge_source = []
        edge_target = []
        for i, j in edge_list:
            edge_source.append(i)
            edge_target.append(j + self.cossim.shape[0])
        # add self loop
        for i in range(self.cossim.shape[0] + self.cossim.shape[1]):
            edge_source.append(i)
            edge_target.append(i)
        self.crosscity_graph = dgl.graph((torch.Tensor(edge_source).long(), torch.Tensor(edge_target).long()))
        self.feat_gatconv = BatchGATConv(in_feats = in_dim, out_feats = 16, num_heads = 4, residual = True)

        self.prediction_linear = nn.Linear(64, 1)
    
    def forward(self, source_feats, target_feats):
        # source_feats: (B, F, N)
        # target_feats: (B, F, N)
        concat_feats = torch.cat([source_feats, target_feats], dim = -1).permute(2, 0, 1).contiguous()
        propagated_feat = F.relu(self.feat_gatconv(self.crosscity_graph, concat_feats).flatten(2))[self.cossim.shape[0]:, :, :]
        predict = self.prediction_linear(propagated_feat).permute(1, 2, 0).contiguous()
        return predict # (B, N, 1)