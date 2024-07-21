import torch
import torch.nn as nn
from .UNet_SA import UNet_SA as SubNN_X
from torch.nn import functional as F
import numpy as np

def grad_op(x):
    res = torch.zeros(*x.shape, 2).type_as(x)
    res[:, :, :, 1:, 0] = x[..., 1:] - x[..., :-1]
    res[:, :, 1:, :, 1] = x[:, :, 1:, :] - x[:, :, :-1, :]
    return res

def grad_T_op(grad_res):
    grad_x, grad_y = grad_res[:, :, :, 1:, 0], grad_res[:, :, 1:, :, 1]
    grad_x_pad, grad_y_pad = F.pad(grad_x, (1, 1, 0, 0)), F.pad(grad_y, (0, 0, 1, 1))
    res = grad_x_pad[..., :-1] - grad_x_pad[..., 1:] + grad_y_pad[:, :, :-1, :] - grad_y_pad[:, :, 1:, :]
    return res

class SubNN_E(nn.Module):
    def __init__(self, in_c, hidden_c, block_n=3, bias=True):
        super(SubNN_E, self).__init__()
        self.blocks = [nn.Conv2d(in_c, hidden_c, 3, 1, 1, bias=bias)]
        for i in range(block_n - 1):
            self.blocks.extend([nn.PReLU(), nn.Conv2d(hidden_c, hidden_c, 3, 1, 1, bias=bias)])
        self.blocks.extend([nn.PReLU(), nn.Conv2d(hidden_c, in_c, 3, 1, 1, bias=bias)])
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x, ts):
        x2 = self.blocks(x)
        ts = torch.ones_like(x2) * ts.view(-1,1,1,1)
        output = torch.where(torch.abs(x2) > ts, -torch.sign(x2) * ts + x2, torch.zeros_like(x2))
        return output

class CAM(nn.Module):
    def __init__(self, in_channels, stage_num, hidden_channels=64):
        super(CAM, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.fc3 = nn.Linear(hidden_channels, stage_num * 3, bias=True)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        num = x.shape[1]
        num = int(num / 3)
        return x[:, 0:num]*0.2, x[:, num:2*num], x[:, 2*num:]*0.1

class StageBlock(nn.Module):
    def __init__(self):
        super(StageBlock, self).__init__()
        self.x_block = SubNN_X(2, 1, 6, 1)
        self.st_block = SubNN_E(2, 32, 5)

    def forward(self, x, a, grad_y, lba, w, d):
        vk, phik, t = x, x, 1
        for _ in range(10):
            D_grad = grad_T_op(grad_op(vk) - (grad_y - a))
            phiknext = vk - lba.view(-1, 1, 1, 1) * D_grad
            tnext = (0.5 * (1 + np.sqrt(1 + 4 * t * t)))
            vk = phiknext + (t - 1) / tnext * (phiknext - phik)
            phik, t = phiknext, tnext

        x=phik
        x_cat = torch.cat((x, grad_T_op(grad_op(x) - (grad_y - a))), dim=1)
        x = x * w.view(-1, 1, 1, 1) + (1 - w.view(-1, 1, 1, 1)) * self.x_block(x_cat)  # + x

        b = grad_y - grad_op(x)
        b = b.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)
        a = self.st_block(b, d).permute(0, 2, 3, 1).unsqueeze(dim=1)
        return x, a

class network(nn.Module):
    def __init__(self, stage_num):
        super(network, self).__init__()
        stages = []
        self.stage_num = stage_num
        for _ in range(stage_num):
            stages.append(StageBlock())
        self.unrolling_stages = nn.ModuleList(stages)
        self.cond_net = CAM(1, stage_num, hidden_channels=128)

    def forward(self, grad_y, cond, x=None, a=None):
        preds_list = []
        lba, w, d = self.cond_net(cond)
        for i in range(self.stage_num):
            x, a = self.unrolling_stages[i](x, a, grad_y, lba[:,i], w[:,i], d[:,i])
            preds_list.append(x)
        return x, preds_list