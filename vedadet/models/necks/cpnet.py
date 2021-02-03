import logging
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from vedacore.modules import (ConvModule, constant_init, kaiming_init)
from vedacore.misc import registry


@registry.register_module('neck')
class CPNet(nn.Module):

    def __init__(self,
                 in_channels,
                 num_levels,
                 out_channel=256,
                 conv_cfg=dict(typename='Conv1d'),
                 norm_cfg=dict(typename='BN1d')):
        super(CPNet, self).__init__()

        self.cp_layers = []
        for i in range(num_levels):
            out_channels = out_channel
            self.cp_layers.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(typename='ReLU', inplace=True))
            )
            in_channels = out_channels

        self.cp_layers = nn.Sequential(*self.cp_layers)

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x):

        outs = [x]
        for i, layer in enumerate(self.cp_layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)
