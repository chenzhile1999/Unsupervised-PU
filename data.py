from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import math
from scipy.stats import norm

def Wrap(x):
    return np.remainder(x+math.pi, np.ones_like(x) * (2*math.pi))-math.pi

def get_Gaussian_Noise(h, w, SNR):
    reqSNR = 10 ** (SNR / 10)
    sigPower = 1
    sigPower = 10 ** (sigPower / 10)
    noisePower = sigPower / reqSNR
    std = np.sqrt(noisePower)
    noise = std * norm.rvs(0, 1, size=(h, w)).astype(np.float32)
    return noise, std

def grad_op(x):
    res = np.zeros((*x.shape, 2))
    res[:, 1:, 0] = x[:, 1:] - x[:, :-1]
    res[1:, :, 1] = x[1:, :] - x[:-1, :]
    return res.astype(np.float32)

class PUData(Dataset):
    def __init__(self, path='', aug=True):
        super(PUData, self).__init__()
        file=h5py.File(path, 'r')
        self.wrapped=file['psi']
        self.gt = file['phi']
        self.noise = file['snr']
        self.aug=aug

    def __len__(self):
        return self.wrapped.shape[0]

    def __getitem__(self, i):
        wrapped = np.array(self.wrapped[i, ...], dtype=np.float32)
        gt = np.array(self.gt[i, ...], dtype=np.float32)

        if self.aug:
            flip = np.random.randint(0, 2)
            if flip:
                axis = np.random.randint(-2, 0)
                wrapped = np.flip(wrapped, axis=axis)
                gt = np.flip(gt, axis=axis)
            else:
                axis=-99
            rotate = np.random.randint(0, 4)
            wrapped = np.rot90(wrapped, k=rotate, axes =(-2, -1))
            gt = np.rot90(gt, k=rotate, axes=(-2, -1))

        noise, std = get_Gaussian_Noise(wrapped.shape[-2], wrapped.shape[-1], self.noise[i])
        noise = Wrap(wrapped+noise)-wrapped
        WGy = Wrap(grad_op(wrapped))
        WGy_plus, WGy_minus = Wrap(grad_op(wrapped+noise)), grad_op(wrapped-noise)
        data_dict = {'Wrapped':torch.from_numpy(wrapped[None,:].copy()), 'gt':torch.from_numpy(gt.copy()[None,:]),
                    'Wrapped_Grad_y':torch.from_numpy(WGy[None,:].copy()), 'std':torch.Tensor([std]).float(),
                    'WGy_plus':torch.from_numpy(WGy_plus[None,:].copy()), 'WGy_minus':torch.from_numpy(WGy_minus[None,:].copy())}
        return data_dict