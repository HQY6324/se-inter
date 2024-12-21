import torch
import torch.nn as nn
class PpiLoss(nn.Module):


    def __init__(self, alpha=None, clamp=False, reduction='mean'):
        super(PpiLoss, self).__init__()
        self.alpha = alpha
        self.clamp = clamp
        self.reduction = reduction
        if isinstance(alpha,(float,int)):
            self.alpha = torch.tensor(alpha,dtype=torch.float32,requires_grad=True)


    def forward(self, Input, Label):

        device = Label.device
        Constant = torch.tensor(1.0,dtype=torch.float32,requires_grad=True).to(device)

        if self.alpha is not None:
            self.alpha = self.alpha.to(device)#
            weight1 = self.alpha*(2-Input)**2
            weight2 = (Constant-self.alpha)*(1+Input)**3
            loss = -Label*torch.log(Input)*weight1 - (Constant-Label)*torch.log(Constant-Input)*weight2
        else:
            loss = -Label*torch.log(Input) - (Constant-Label)*torch.log(Constant-Input)

        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss