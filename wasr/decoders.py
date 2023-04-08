import torch
from torch import nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import timm
from .layers import AttentionRefinementModule, FeatureFusionModule, ASPPv2, Spade, C3Blk, ResBlk, SPP, Conv, SpatialAttentionRefinementModule, PyramidPoolAgg, SIM, SegHead
from .metaformer import *

class NoIMUDecoder(nn.Module):
    """Decoder without IMU input."""
    def __init__(self, num_classes=3):
        super(NoIMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512, last_arm=True),
            nn.Conv2d(512, 2048, 1) # Equalize number of features with ARM1
        )

        self.ffm = FeatureFusionModule(256, 2048, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], num_classes)

    def forward(self, x, aux, skip2, skip1, imu_mask):
        
        arm1 = self.arm1(x)
        arm2 = self.arm2(skip2)
        arm_combined = arm1 + arm2

        x = self.ffm(skip1, arm_combined)

        output = self.aspp(x)

        return output


class IMUDecoder(nn.Module):
    """Decoder with IMU information merging."""
    def __init__(self, num_classes=3):
        super(IMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048 + 1)
        self.aspp1 = ASPPv2(2048, [6, 12, 18], 32)
        self.ffm1 = FeatureFusionModule(2048 + 1, 32, 1024)

        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512 + 1, last_arm=True),
            nn.Conv2d(512 + 1, 1024, 1, bias=False) # Equalize number of features with FFM1
        )

        self.ffm = FeatureFusionModule(256 + 1, 1024, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], num_classes)
        self.upsample1 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)


    def forward(self, out, aux, skip2, skip1, imu_mask):

        # resize IMU mask to two required scales
        imu_mask_s1 = TF.resize(imu_mask, (out.size(2), out.size(3)), InterpolationMode.NEAREST)
        imu_mask_s0 = TF.resize(imu_mask, (skip1.size(2), skip1.size(3)), InterpolationMode.NEAREST)

        # feature mixer
        out_imu = torch.cat([out, imu_mask_s1], dim=1)
        arm1 = self.arm1(out_imu)
        aspp1 = self.aspp1(out)
        ffm1 = self.ffm1(arm1, aspp1)

        # enriched skip connection
        skip2_imu = torch.cat([skip2, imu_mask_s1], dim=1)
        arm2 = self.arm2(skip2_imu)

        # decoder
        arm_combined = ffm1 + arm2
        skip1_imu = torch.cat([skip1, imu_mask_s0], dim=1)
        x = self.ffm(skip1_imu, arm_combined)

        # head
        output = self.aspp(x)

        return output


class IMUDecoderSmall(nn.Module):
    """WaSR-light IMU decoder."""
    def __init__(self, num_classes=3, ch = [512, 256, 128, 64]):
        super(IMUDecoderSmall, self).__init__()
    
        self.arm1 = AttentionRefinementModule(ch[0] + 1)
        self.aspp1 = ASPPv2(ch[0], [6, 12, 18], 32)
        self.ffm1 = FeatureFusionModule(ch[0] + 1, 32, ch[1])

        self.arm2 = nn.Sequential(
                AttentionRefinementModule(ch[2] + 1, last_arm=True),
                nn.Conv2d(ch[2] + 1, ch[1], 1, bias=False) # Equalize number of features with FFM1
            )

        self.ffm = FeatureFusionModule(ch[3] + 1, ch[1], ch[1])
        self.aspp = ASPPv2(ch[1], [6, 12, 18, 24], num_classes)
        
        self.upsample= nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        
    def forward(self, out, aux, skip2, skip1, imu_mask):

        # resize IMU mask to the required scales
        imu_mask_s2 = TF.resize(imu_mask, (out.size(2), out.size(3)), InterpolationMode.NEAREST)
        imu_mask_s1 = TF.resize(imu_mask, (skip2.size(2), skip2.size(3)), InterpolationMode.NEAREST)
        imu_mask_s0 = TF.resize(imu_mask, (skip1.size(2), skip1.size(3)), InterpolationMode.NEAREST)
        
        # feature mixer
        out_imu = torch.cat([out, imu_mask_s2], dim=1)
        arm1 = self.arm1(out_imu)
        aspp1 = self.aspp1(out)
        ffm1 = self.ffm1(arm1, aspp1)

        # enriched skip connection
        skip2_imu = torch.cat([skip2, imu_mask_s1], dim=1)
        arm2 = self.arm2(skip2_imu)

        # decoder
        ffm1_up = self.upsample(ffm1)
        arm_combined = ffm1_up + arm2
        skip1_imu = torch.cat([skip1, imu_mask_s0], dim=1)
        arm_combined_up = self.upsample1(arm_combined)
        x = self.ffm(skip1_imu, arm_combined_up)

        # head
        output = self.aspp(x)

        return output

def get_token_mixer(letter):
    if letter == "A":
        return Attention
    elif letter == "P":
        return Pooling
    elif letter == "C":
        return ARMMixer
    elif letter == "S":
        return SpatialAttentionMixer
    else:
        raise ValueError(f"Mixer {mixer} not supported.")


class EWaSRDecoder(nn.Module):
    
    def __init__(self, num_classes=3, L = 6, ch=512, ch_sim=256, mixer="CCCCSS", enricher="SS", imu=False, project=False):
        
        super(EWaSRDecoder, self).__init__()
        
        self.project = project
        self.ch = [ch, ch//2, ch//4, ch//8] if isinstance(ch, int) else ch

        if self.project:
            self.convs_project1 = nn.Conv2d(self.ch[0], self.ch[0] // 2, 1)
            self.convs_project2 = nn.Conv2d(self.ch[1], self.ch[1] // 2, 1)
            self.convs_project3 = nn.Conv2d(self.ch[2], self.ch[2] // 2, 1)
            self.convs_project4 = nn.Conv2d(self.ch[3], self.ch[3] // 2, 1)
            self.ch = [ch // 2 for ch in self.ch]

        self.imu = imu
        self.pool_feats = PyramidPoolAgg(2)        
        self.metaformers = None
        self.metaformers_skip2 = None

        if len(mixer) > 0:
            metaformers = []
            for letter in mixer:
                metaformers.append(MetaFormerBlock(sum(self.ch), token_mixer=get_token_mixer(letter)))
            self.metaformers = nn.Sequential(*metaformers)

        if len(enricher) > 0:
            metaformers_skip2 = []
            for letter in enricher:
                metaformers_skip2.append(MetaFormerBlock(self.ch[2], token_mixer=get_token_mixer(letter)))
            self.metaformers_skip2 = nn.Sequential(*metaformers_skip2)

        """
        if mix:
            metaformers = [(
                MetaFormerBlock(sum(self.ch), token_mixer=ARMMixer),
                MetaFormerBlock(sum(self.ch), token_mixer=SpatialAttentionMixer))
                 for _ in range(L//2)]
            metaformers = [item for sublist in metaformers for item in sublist]
        elif short is None:
            metaformers = [MetaFormerBlock(sum(self.ch), token_mixer=token_mixer) for _ in range(L)]
        else:
            metaformers = [MetaFormerBlock(sum(self.ch), token_mixer=token_mixer) for _ in range(L-2)]
            short_mixer = SpatialAttentionMixer if short == "sarm" else ARMMixer
            metaformers.extend([
                MetaFormerBlock(sum(self.ch), token_mixer=short_mixer),
                MetaFormerBlock(sum(self.ch), token_mixer=short_mixer)
            ])
        self.metaformers = nn.Sequential(*metaformers)

        if mix:
            metaformers_skip2 = [MetaFormerBlock(self.ch[2], token_mixer=ARMMixer), MetaFormerBlock(self.ch[2], token_mixer=SpatialAttentionMixer)]
            self.metaformers_skip2 = nn.Sequential(*metaformers_skip2)
        elif skip is not None:
            token_mixer_skip2 = SpatialAttentionMixer if skip == "sarm" else ARMMixer
            metaformers_skip2 = [MetaFormerBlock(self.ch[2], token_mixer=token_mixer_skip2) for _ in range(2)]
            self.metaformers_skip2 = nn.Sequential(*metaformers_skip2)
        """

        self.sim1 = SIM(self.ch[0], ch_sim)
        self.sim2 = SIM(self.ch[1], ch_sim)
        self.sim3 = SIM(self.ch[2], ch_sim)
        self.sim4 = SIM(self.ch[3], ch_sim)
        
        head_in = ch_sim+1 if self.imu else ch_sim
        self.seg_head = SegHead(head_in, num_classes)

        
    def forward(self, x, aux, skip2, skip1, imu_mask):

        if self.project:
            x = self.convs_project1(x)
            aux = self.convs_project2(aux)
            skip2 = self.convs_project3(skip2)
            skip1 = self.convs_project4(skip1)
        
        tokens = self.pool_feats([x, aux, skip2, skip1])

        if self.metaformers is not None:
            tokens = self.metaformers(tokens)

        if self.metaformers_skip2 is not None:
            skip2 = self.metaformers_skip2(skip2)
        
        f1 = self.sim1(x, tokens[:, :self.ch[0], :, :])
        f2 = self.sim2(aux, tokens[:, self.ch[0]:sum(self.ch[:2]), :, :])
        f3 = self.sim3(skip2, tokens[:, sum(self.ch[:2]):sum(self.ch[:3]), :, :])
        f4 = self.sim4(skip1, tokens[:, sum(self.ch[:3]):, :, :])
        
        
        f1 = TF.resize(f1, (f4.size(2), f4.size(3)), InterpolationMode.BILINEAR)
        f2 = TF.resize(f2, (f4.size(2), f4.size(3)), InterpolationMode.BILINEAR)
        f3 = TF.resize(f3, (f4.size(2), f4.size(3)), InterpolationMode.BILINEAR)
           
        x = f1 + f2 + f3 + f4

        if self.imu: 
            imu_mask = TF.resize(imu_mask, (x.size(2), x.size(3)), InterpolationMode.NEAREST)
            x = torch.cat([x, imu_mask], dim=1)
        
        x = self.seg_head(x)
        
        return x