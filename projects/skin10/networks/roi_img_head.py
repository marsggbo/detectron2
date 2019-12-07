import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from detectron2.utils.registry import Registry

from .utils import create_maskimg

ROI_IMG_HEAD_REGISTRY = Registry("ROI_IMG_HEAD")
ROI_IMG_HEAD_REGISTRY.__doc__ = """"""


def build_roi_img_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_IMG_HEAD.NAME
    return ROI_IMG_HEAD_REGISTRY.get(name)(cfg, input_shape)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


@ROI_IMG_HEAD_REGISTRY.register()
class RoiImgHead_Resnet(nn.Module):
    def __init__(self, cfg, input_shape=None):
        '''
        params:
            pretrained: load the pretrained models of resnet
            model_root: the path of pretrained weights of resnet
        '''
        super(RoiImgHead_Resnet, self).__init__()
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.softmask_theta = cfg.MODEL.SOFT_MASK.THETA

        # name = cfg.MODEL.ROI_IMG_HEAD.TYPE
        name = 'resnet50'
        assert name in ['resnet18', 'resnet50', 'resnet101']
        backbone = eval(f"models.{name}()")
        backbone.fc = Identity()
        backbone = list(backbone.children())
        self.stem = nn.Sequential(*backbone[:4])
        self.layer1 = nn.Sequential(*backbone[4])
        self.layer2 = nn.Sequential(*backbone[5])
        self.layer3 = nn.Sequential(*backbone[6])
        self.layer4 = nn.Sequential(*backbone[7])
        
        self.clf = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def losses(self, preds, gt_classes):
        '''
        preds (FloatTensor): N*C
        gt_classes (LongTensor): N 
        '''
        return {
            'loss_img_cls': F.cross_entropy(preds, gt_classes, reduction="mean")
        }

    def forward(self, x, gt_classes, proposals=None):
        '''
        params: 
            x (tensor): N*c*h*w
            gt_classes (tensor): N
            proposals (): 

        return:
            img_cls_preds (tensor): N*classes
            losses (dict): {'loss_img_cls': loss: tensor}
        '''
        bs= x.shape[0]
        new_x = create_maskimg(x, proposals) if proposals else x
        features = self.extract_features(new_x)
        img_cls_preds = self.clf(features).view(bs, -1)

        losses = self.losses(img_cls_preds, gt_classes)
        if not self.training:
            return {
                'img_cls_preds': img_cls_preds,
                # 'loss': losses
            }
        return img_cls_preds, losses

    def extract_features(self, x):
        features = {}
        features['stem'] = self.stem(x)
        features['layer1'] = self.layer1(features['stem'])
        features['layer2'] = self.layer2(features['layer1'])
        features['layer3'] = self.layer3(features['layer2'])
        features['layer4'] = self.layer4(features['layer3'])
        final_feature = features['layer4']
        return final_feature

    def inference(self, x, boxes):
        pass

if __name__ == "__main__":
    class CFG:
        num_classes = 10
    cfg = CFG()
    net = RoiImgHead_Resnet(cfg)
    x = torch.rand(2,3,260,254)
    y = net(x)
