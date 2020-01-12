import torch
from torch import nn
import torch.nn.functional as F


class GHM_Loss(nn.Module):
    def __init__(self, bins, momentum):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._momentum = momentum
        self._last_bin_count = None

        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if self._momentum > 0:
            self.acc_sum = [0.] * bins

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        batch_size, cls_size = x.shape

        beta = torch.zeros((batch_size,)).cuda()

        # gradient length
        labels = torch.stack([1 - target.clone(), target], dim=-1).detach()
        g = torch.abs(self._custom_loss_grad(x, labels)).detach()

        n = 0
        for i in range(self._bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                self.acc_sum[i] = self._momentum * self.acc_sum[i] + (1 - self._momentum) * num_in_bin
                beta[inds] = batch_size / self.acc_sum[i]
                n += 1
        if n > 0:
            beta = beta / n

        beta = beta.detach()

        return self._custom_loss(x, target, beta)


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins, momentum):
        super(GHMC_Loss, self).__init__(bins, momentum)

    def _custom_loss(self, x, target, beta):
        loss =  beta*F.nll_loss(torch.log(x), target,reduction='none')
        return loss.mean()

    def _custom_loss_grad(self, x, target):
        return torch.mean(target * (x.detach() - target), dim=-1)


class GHMR_Loss(GHM_Loss):
    ''' 
    TODO(jiandong.ye): not implement
    '''

    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)
