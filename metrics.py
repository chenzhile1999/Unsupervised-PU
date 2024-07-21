import torch
import math

def torch_Wrap(x):
    return torch.remainder(x+math.pi, torch.ones_like(x) * (2*math.pi))-math.pi

def NRMSE_batch(gt, out):
    gt=gt.squeeze(1)
    out=out.squeeze(1)
    gt_min, gt_max = gt.flatten(start_dim=1).min(dim=-1, keepdim=True).values.unsqueeze(-1), gt.flatten(start_dim=1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    out_min, out_max = out.flatten(start_dim=1).min(dim=-1, keepdim=True).values.unsqueeze(-1), out.flatten(start_dim=1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    out_scaled = (out - out_min) / (out_max - out_min) * (gt_max-gt_min) + gt_min
    error = gt - out_scaled
    r = gt_max-gt_min
    nrmse = torch.sqrt(torch.mean(error ** 2, dim=(-2,-1))) / r.squeeze() * 100
    return nrmse, out_scaled

def gradient(x):
    res = torch.zeros(*x.shape, 2).type_as(x)
    res[:, :, :, 1:, 0] = x[..., 1:] - x[..., :-1]
    res[:, :, 1:, :, 1] = x[:, :, 1:, :] - x[:, :, :-1, :]
    return res

def Loss_SD(target, pred):
    pred_grad = gradient(pred)
    target_grad = gradient(target)
    loss = torch.mean(torch.abs(target_grad-pred_grad))
    return loss

def Loss_SR(target_grad, pred):
    pred_grad = gradient(pred)
    error = torch_Wrap(target_grad - pred_grad)
    loss = torch.mean(torch.square(error))
    return loss