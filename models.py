import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class CNN(torch.nn.Module):
    def __init__(self, nc, nfm, img_size):
        super(CNN, self).__init__()

        exp = int( math.log(img_size)/math.log(2) )

        self.cnn = [spectral_norm(nn.Conv2d(nc, nfm, 4, 2, 1)),
                   nn.ReLU()]

        for i in range(exp-3):
          self.cnn += [spectral_norm(nn.Conv2d( nfm*(2**i) , nfm*( 2**(i+1) ), 4, 2, 1)),
                      nn.ReLU()]

        self.cnn += [spectral_norm(nn.Conv2d( nfm*( 2**(exp-3) ) , 1, 4, 1, 0)),
                    nn.Sigmoid()]

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self, inputs):
        return self.cnn(inputs)

class ResidualBlock(torch.nn.Module):
    def __init__(self, nfm):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
          spectral_norm(nn.Conv2d(nfm, nfm, 3, 1, 1)),
          nn.BatchNorm2d(nfm),
          nn.ReLU(),
          spectral_norm(nn.Conv2d(nfm, nfm, 3, 1, 1)),
          nn.BatchNorm2d(nfm)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        result = x + self.conv_block(x)
        out = self.relu(result)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, nfm, layers):
        super(ResNet, self).__init__()

        self.resnet = []
        for _ in range(layers):
            self.resnet += [ResidualBlock(nfm)]

        self.resnet = nn.Sequential(*self.resnet)

    def forward(self, x):
        return self.resnet(x)


class U_Net_Block(torch.nn.Module):
    def __init__(self, submodule, nfm=64, in_frames=None, out_frames=None, outermost=False, innermost=False):
        super(U_Net_Block, self).__init__()

        self.outermost = outermost
        self.innermost = innermost
        self.nfm = nfm

        if outermost:
            in_down_nc = in_frames*3
            out_up_nc = out_frames*3
        else:
            in_down_nc = nfm
            out_up_nc = nfm

        downconv_1 = spectral_norm(nn.Conv2d(in_down_nc, in_down_nc, 3, 1, 1))
        downnorm_1 = nn.BatchNorm2d(in_down_nc)

        downconv = spectral_norm(nn.Conv2d(in_down_nc, nfm*2, 3, 2, 1))
        downnorm = nn.BatchNorm2d(nfm*2)

        upconv_1 = nn.ConvTranspose2d(nfm*4, nfm*4, 3, 1, 1)
        upnorm_1 = nn.BatchNorm2d(nfm*4)

        upconv = nn.ConvTranspose2d(nfm*4, out_up_nc*2, 3, 2, 0)
        upnorm = nn.BatchNorm2d(out_up_nc*2)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        if outermost:
            upconv_1 = nn.ConvTranspose2d(nfm*4, nfm*4, 3, 1, 1)
            upnorm = nn.BatchNorm2d(nfm*4)

            upconv = nn.ConvTranspose2d(nfm*4, out_up_nc, 3, 2, 1)

            down = [downconv_1, downnorm_1, relu, downconv, downnorm, relu]
            up = [upconv_1, upnorm, relu, upconv, tanh]
        elif innermost:
            res_nfm = 64

            downconv_1 = spectral_norm(nn.Conv2d(nfm, nfm, 3, 1, 1))
            downnorm_1 = nn.BatchNorm2d(nfm)

            downconv = spectral_norm(nn.Conv2d(nfm, res_nfm, 3, 2, 1))
            downnorm = nn.BatchNorm2d(res_nfm)

            upconv_1 = nn.ConvTranspose2d(res_nfm, res_nfm, 3, 1, 1)
            upnorm_1 = nn.BatchNorm2d(res_nfm)

            upconv = nn.ConvTranspose2d(res_nfm, out_up_nc, 3, 2, 0)
            upnorm = nn.BatchNorm2d(out_up_nc)

            down = [downconv_1, downnorm_1, relu, downconv, downnorm, relu]
            up = [upconv_1,  upnorm_1, relu, upconv, upnorm, relu]
        else:
            downconv_1 = spectral_norm(nn.Conv2d(in_down_nc*2, in_down_nc*2, 3, 1, 1))
            downnorm_1 = nn.BatchNorm2d(in_down_nc*2)

            downconv =spectral_norm( nn.Conv2d(in_down_nc*2, nfm*2, 3, 2, 1))

            down = [downconv_1, downnorm_1, relu, downconv, downnorm, relu]
            up = [upconv_1, upnorm_1, relu, upconv, upnorm, relu]

        self.u_block = down + [submodule] + up
        self.u_block = nn.Sequential(*self.u_block)

    def forward(self, x):
        if self.outermost:
            return self.u_block(x)
        elif not self.innermost:
            hidden_idx = int(np.log2(1024//x.shape[2])-1)
            cat_recurrent = torch.cat([self.hidden[hidden_idx], x], 1)
            u_out = self.u_block(cat_recurrent)
            u_out = nnf.interpolate(u_out, size=(x.shape[2], x.shape[2]), mode='bilinear', align_corners=False)
            cat_feature_layers = torch.cat([x, u_out], 1)
            self.hidden = self.hidden[:hidden_idx] + [ cat_feature_layers[:, :x.shape[1], :, :] ] + self.hidden[hidden_idx+1:]

            return cat_feature_layers[:, x.shape[1]:, :, :]
        else:
            cat_feature_layers = torch.cat([x, self.u_block(x)[:, :, :32, :32]], 1)

            return cat_feature_layers

    def _init_hidden(self, batch_size):
        if torch.cuda.is_available(): self.hidden = [torch.zeros(batch_size, self.nfm, 512, 512).cuda(),
                                                     torch.zeros(batch_size, self.nfm, 256, 256).cuda(),
                                                     torch.zeros(batch_size, self.nfm, 128, 128).cuda(),
                                                     torch.zeros(batch_size, self.nfm, 64, 64).cuda()]
        else: self.hidden = [torch.zeros(batch_size, self.nfm, 512, 512),
                             torch.zeros(batch_size, self.nfm, 256, 256),
                             torch.zeros(batch_size, self.nfm, 128, 128),
                             torch.zeros(batch_size, self.nfm, 64, 64)]


class U_Net(torch.nn.Module):
    def __init__(self, nfm, base_network, in_frames, out_frames):
        super(U_Net, self).__init__()

        # ResNet Feature Maps: Res_Nfm x 16 x 16
        self.unet_in = U_Net_Block(base_network, nfm*16, innermost=True)
        # Unet: nfm*4 x 32 x 32
        self.unet_mid4 = U_Net_Block(self.unet_in, nfm*8)
        # Unet: nfm*2 x 64 x 64
        self.unet_mid3 = U_Net_Block(self.unet_mid4, nfm*4)
        # Unet: nfm*4 x 128 x 128
        self.unet_mid2 = U_Net_Block(self.unet_mid3, nfm*2)
        # Unet: nfm*2 x 256 x 256
        self.unet_mid1 = U_Net_Block(self.unet_mid2, nfm)
        # Unet: nfm x 512 x 512
        self.unet = U_Net_Block(self.unet_mid1, int(nfm/2), in_frames=in_frames, out_frames=out_frames, outermost=True)
        # Unet: num_frames x 1024 x 1024

    def forward(self, x):
        return self.unet(x)

    def _init_hidden(self, batch_size):
        self.unet_mid1._init_hidden(batch_size)
        self.unet_mid2._init_hidden(batch_size)
        self.unet_mid3._init_hidden(batch_size)
        self.unet_mid4._init_hidden(batch_size)
