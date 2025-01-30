from . import resnext, AMSoftmaxLoss
from torch import nn


class ResNeXt_with_AMSoftmaxLoss(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXt_with_AMSoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.net = resnext.resnext101_32x8d(pretrained=False)
        self.loss = AMSoftmaxLoss.AMSMLoss(
            feat_dim=2048, n_classes=self.num_classes)

    def forward(self, x, labels, embed=False):
        _, features = self.net(x)
        if not embed:
            loss, pred, scores = self.loss(features, labels)
            return loss, pred, scores
        else:
            return features
