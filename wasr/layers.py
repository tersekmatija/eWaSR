import torch
from torch import nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, last_arm=False):
        super(AttentionRefinementModule, self).__init__()

        self.last_arm = last_arm

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x

        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        weights = self.sigmoid(x)

        out = weights * input

        if self.last_arm:
            weights = self.global_pool(out)
            out = weights * out

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, bg_channels, sm_channels, num_features):
        super(FeatureFusionModule, self).__init__()

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_features, num_features, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_big, x_small):
        if x_big.size(2) > x_small.size(2):
            x_small = self.upsampling(x_small)

        x = torch.cat((x_big, x_small), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_out = self.relu(x)

        x = self.global_pool(conv1_out)
        x = self.conv2(x)
        x = self.conv3(x)
        weights = self.sigmoid(x)

        mul = weights * conv1_out
        out = conv1_out + mul

        return out


class ASPPv2Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bias=False, bn=False, relu=False):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=bias))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))

        if relu:
            modules.append(nn.ReLU())

        super(ASPPv2Conv, self).__init__(*modules)


class ASPPv2(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, relu=False, biased=True):
        super(ASPPv2, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPv2Conv(in_channels, out_channels, rate, bias=True))

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Sum convolution results
        res = torch.stack(res).sum(0)
        return res


class Spade(nn.Module):
    
    def __init__(self, in_channels, feat_channels, kernel_size, num_features = 128, norm_type = "sync", use_relu6 = True):
        super().__init__()
        
        if norm_type == "sync":
            self.normalize = nn.BatchNorm2d(in_channels)
        elif norm_type == "instance":
            self.normalize = nn.InstanceNorm2d(in_channels)
        elif norm_type == "batch":
            self.normalize = nn.BatchNorm2d(in_channels)
        else:
            raise ValueError(f"Not supported normalization in SPADE: {norm_type}. Supported: sync, instance, batch")
                             
        self.conv = nn.Sequential(
            nn.Conv2d(feat_channels, num_features, kernel_size, padding = kernel_size // 2),
            nn.ReLU6() if use_relu6 else nn.ReLU() # add optional ReLU6 for low precision computation
        )
                             
        self.conv_gamma = nn.Conv2d(num_features, in_channels, kernel_size, padding = kernel_size // 2)
        self.conv_beta = nn.Conv2d(num_features, in_channels, kernel_size, padding = kernel_size // 2)
                
    def forward(self, x, feat):
        feat = self.conv(feat)
        gamma = self.conv_gamma(feat)
        beta = self.conv_beta(feat)
        
        x = self.normalize(x)
        
        x = x * (1 + gamma) + beta
        
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1, g=g, p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class NonBottleneck1DBlk(nn.Module):
    # ESANet / ERFNet (https://arxiv.org/pdf/2011.06961.pdf)
    def __init__(self, c1, c2, shortcut=True, spade=True):
        super(NonBottleneck1DBlk, self).__init__()
        self.spade = spade
        self.add = shortcut and c1 == c2

        self.cv1 = nn.Conv2d(c1, c2, kernel_size = (3,1), padding=(1,0))
        self.relu1 = nn.ReLU()

        self.cv2 = nn.Conv2d(c2, c2, kernel_size = (1,3), padding=(0,1))
        self.relu2 = nn.ReLU()
        self.spade2 = Spade(c2, 1, 3) if spade else nn.BatchNorm2d(c2)

        self.cv3 = nn.Conv2d(c2, c2, kernel_size = (3,1), padding=(1,0))
        self.relu3 = nn.ReLU()

        self.cv4 = nn.Conv2d(c2, c2, kernel_size = (1,3), padding=(0,1))
        self.relu4 = nn.ReLU()
        self.spade4 = Spade(c2, 1, 3) if spade else nn.BatchNorm2d(c2)

    def forward(self, x, feats):
        y = self.cv1(x)
        y = self.relu1(y)

        y = self.cv2(y)
        y = self.relu2(y)
        y = self.spade2(y, feats) if self.spade else self.spade2(y)

        y = self.cv3(y)
        y = self.relu3(y)

        y = self.cv4(y)
        y = self.relu4(y)
        y = self.spade4(y, feats) if self.spade else self.spade4(y)

        return x + y if self.add else y

 
class ResBlk(nn.Module):
    
    def __init__(self, c1, c2, shortcut=True, spade=True):
        super(ResBlk, self).__init__()
        
        self.add = shortcut and c1 == c2
        self.spade = spade
        
        self.spade1 = Spade(c1, 1, 3) if spade else nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(c1, c2, 3, padding = 1)
        
        self.spade2 = Spade(c2, 1, 3) if spade else nn.BatchNorm2d(c1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(c2, c2, 3, padding = 1)
        
    def forward(self, x, feats):
        y = self.spade1(x, feats) if self.spade else self.spade1(x)
        y = self.relu1(y)
        y = self.conv1(y)
        y = self.spade2(y, feats) if self.spade else self.spade2(y)
        y = self.relu2(y)
        y = self.conv2(y)
        
        return x + y if self.add else y


class C3Blk(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, shortcut=True, spade=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        g = 1
        n = 1
        self.spade = spade
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(2 * c2, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c2, c2, shortcut, g) for _ in range(n)))
        self.spade1 = Spade(c2*2, 1, 3) if spade else nn.Identity()

    def forward(self, x, feats):
        x = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        #print(x.size())
        x = self.spade1(x, feats) if self.spade else self.spade1(x)
        x = torch.nn.functional.relu(x) # TODO: remove this
        x = self.cv3(x)
        return x


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SpatialAttentionRefinementModule(nn.Module):
    
    def __init__(self, in_channels, ks = 7):
        super(SpatialAttentionRefinementModule, self).__init__()
        g = 1
        n = 1
        p = ks//2
        
        self.conv = nn.Conv2d(3, 1, ks, padding=p, bias=True) # TODO: try false
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, imu_mask):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        
        w = torch.cat([x1, x2, imu_mask], dim=1)
        w = self.conv(w)
        return x * self.sigmoid(w)


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[0].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

    
class SIM(nn.Module):
    def __init__(self, ch_in, ch_out):
        
        super(SIM, self).__init__()
        
        self.conv_lc = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_lc = nn.BatchNorm2d(ch_out)
        
        self.conv_gc = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_gc = nn.BatchNorm2d(ch_out)
        self.sigmoid_gc = nn.Sigmoid()
        
        self.conv_gc1 = nn.Conv2d(ch_in, ch_out, 1)
        self.bn_gc1 = nn.BatchNorm2d(ch_out)
        
    def forward(self, lx, gx):
        
        gx = TF.resize(gx, (lx.size(2), lx.size(3)), InterpolationMode.NEAREST)

        lx = self.bn_lc(self.conv_lc(lx))
        multi = self.sigmoid_gc(self.bn_gc(self.conv_gc(gx)))
        gx = self.bn_gc1(self.conv_gc1(gx))
        
        return lx * multi + gx
    

class SegHead(nn.Module):
    
    def __init__(self, ch, num_classes):
        
        super(SegHead, self).__init__()
        
        self.conv = nn.Conv2d(ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU6(ch)
        self.conv1 = nn.Conv2d(ch, num_classes, 1)
        
    def forward(self, x):
        
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv1(x)
        return x