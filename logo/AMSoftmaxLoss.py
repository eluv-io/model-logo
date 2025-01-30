import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger
"""
    Found a cleaner version than mine, both are working, I will keep this cleaner one
"""
class AMSMLoss(nn.Module):

    def __init__(self, feat_dim, n_classes, eps=1e-7, s=30, m=0.35):
        super(AMSMLoss, self).__init__()
        
        self.s = s
        self.m = m
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.fc = nn.Linear(feat_dim, n_classes, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.n_classes
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        
        # manually compute softmax loss, can not get the prediction through this
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)   # (true label's cos_theta - m) * s for each sample in batch   shape is [B, ]
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0) # exclude the target label distance
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)

        scores, pred = torch.max(wf, dim=1)
        return -torch.mean(L), pred, scores


if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    net = AMSMLoss(2048, 1000).to("cuda")
    from torchsummary import summary
    demo_feat = torch.rand((3, 2048)).to("cuda")
    demo_label = torch.zeros((3, ), dtype=int).to("cuda")

    l, pred = net(demo_feat, demo_label)
    logger.info(l.shape, pred.shape)
    logger.info(l, pred)
