import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class Down(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mode='pool'):
        super(Down, self).__init__()
        if mode == 'pool':
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.down = nn.Sequential(*[nn.Conv2d(in_ch, out_ch, 3,2,1), nn.PReLU()])

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mode='upsample'):
        super(Up, self).__init__()
        if mode == 'upsample':
            self.up = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, out_ch, 3,1,1), nn.PReLU()])
        else:
            self.up = nn.Sequential(*[nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1), nn.PReLU()])
    def forward(self, x):
        return self.up(x)

class UNet_SA(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, blk_num=4):
        super(UNet_SA, self).__init__()
        self.start = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True))
        self.down1 = Down(base_ch * 1, base_ch * 1, 'pool')
        self.cbr1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=(3, 3), padding=1),
            nn.PReLU(),
        )

        self.down2 = Down(base_ch * 2, base_ch * 2, 'pool')
        self.cbr2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=(3, 3), padding=1),
            nn.PReLU(),
        )

        self.down3 = Down(base_ch * 4, base_ch * 4, 'pool')
        self.cbr3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=(3, 3), padding=1),
            nn.PReLU(),
        )

        self.down4 = Down(base_ch * 8, base_ch * 8, 'pool')
        self.cbr4 = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=(3, 3), padding=1),
            nn.PReLU(),
        )

        self.blk_num = blk_num
        self.blks = nn.ModuleList([SABlock(fea_dim=base_ch * 16, head_num=8) for i in range(blk_num)])


        self.up4 = Up(base_ch * 16, base_ch * 8, 'upsample')
        self.fuse_bn_act4 = nn.Sequential(
            nn.Conv2d(base_ch * 16, base_ch * 8, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.up3 = Up(base_ch * 8, base_ch * 4, 'upsample')
        self.fuse_bn_act3 = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.up2 = Up(base_ch * 4, base_ch * 2, 'upsample')
        self.fuse_bn_act2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        self.up1 = Up(base_ch * 2, base_ch * 1, 'upsample')
        self.fuse_bn_act1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 1, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.tail_conv = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1, bias=True)


    def forward(self, x):
        x_start = self.start(x)

        d1 = self.down1(x_start)
        f1 = self.cbr1(d1) #2

        d2 = self.down2(f1)
        f2 = self.cbr2(d2) #4

        d3 = self.down3(f2)
        f3 = self.cbr3(d3) #8

        d4 = self.down4(f3)
        f4 = self.cbr4(d4) #16

        x = positional_encoding_2d_as(f4) + f4
        for i in range(self.blk_num):
            x = self.blks[i](x)

        u4 = self.up4(x) #8
        cat4 = torch.cat([u4, f3], dim=1)
        x = self.fuse_bn_act4(cat4)

        u3 = self.up3(x) #4
        cat3 = torch.cat([u3, f2], dim=1)
        x = self.fuse_bn_act3(cat3)

        u2 = self.up2(x) #2
        cat2 = torch.cat([u2, f1], dim=1)
        x = self.fuse_bn_act2(cat2)

        u1 = self.up1(x) #1
        cat1 = torch.cat([u1, x_start], dim=1)
        x = self.fuse_bn_act1(cat1)
        out = self.tail_conv(x)
        return out

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1 or classname.find('ConvTranspose2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

class SABlock(nn.Module):
    '''
    Multi-head attention + Feed-forward network
    '''

    def __init__(self, fea_dim=128, head_num=4):
        super(SABlock, self).__init__()
        self.fea_dim = fea_dim
        self.head_num = head_num
        self.q = nn.Linear(fea_dim, fea_dim)
        self.kv = nn.Linear(fea_dim, fea_dim * 2)
        self.proj = nn.Linear(fea_dim, fea_dim)
        self.norm = nn.LayerNorm(fea_dim)
        self.head_dim = fea_dim // head_num
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        '''

        :param x: B, C, H, W
        :return: b, c, h, w
        '''
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # b, hw, c
        nx = self.norm(x)
        q = self.q(nx)
        k, v = torch.chunk(self.kv(nx), 2, -1)

        q = q.reshape(b, h * w, self.head_num, c // self.head_num).transpose(1, 2)
        k = k.reshape(b, h * w, self.head_num, c // self.head_num).transpose(1, 2)
        v = v.reshape(b, h * w, self.head_num, c // self.head_num).transpose(1, 2)

        x = self.proj(attention(q, k, v, scale=self.scale).transpose(1, 2).flatten(-2, -1)) + x
        x = x.unflatten(1, (h, w)).permute(0, 3, 1, 2)
        return x


@torch.jit.script
def coordinate_encoding_2d(shape: Tuple[int, int, int], scale: float = 2 * math.pi,
                           dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    """Returns the two-dimensional positional encoding as shape [d_model, h, w]"""
    d_model, h, w = shape[-3:]
    X, Y = torch.meshgrid(torch.arange(0, h) / float(h - 1), torch.arange(0, w) / float(w - 1))
    u = torch.stack((X, Y), dim=0).repeat(d_model//2,1,1) * 2 - 1
    return u  # with channel format: sin(x0) sin(y0) cos(x0) cos(y0) sin(x1) ...

@torch.jit.script
def coordinate_encoding_2d_as(x: torch.Tensor, scale: float = 2 * math.pi):
    d, h, w = x.shape[-3:]
    return coordinate_encoding_2d((d, h, w), scale, x.dtype, x.device).type_as(x).expand_as(x)


@torch.jit.script
def positional_encoding_2d(shape: Tuple[int, int, int], temperature: float = 1e4, scale: float = 2 * math.pi,
                           dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    """Returns the two-dimensional positional encoding as shape [d_model, h, w]"""
    d_model, h, w = shape[-3:]
    i = torch.arange(d_model // 4, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device) / (h - 1) * scale
    xs = torch.arange(w, dtype=dtype, device=device) / (w - 1) * scale
    t = (temperature ** (4. / d_model * i)).view(-1, 1, 1, 1, 1).expand(-1, 2, -1, -1, -1)
    u = torch.cat((xs.expand(1, h, w), ys.unsqueeze(-1).expand(1, h, w)), -3) / t
    u2 = u.clone()
    u2[:, 0] = u[:, 0].sin()
    u2[:, 1] = u[:, 1].cos()
    return u2.reshape(-1, h, w)  # with channel format: sin(x0) sin(y0) cos(x0) cos(y0) sin(x1) ...

@torch.jit.script
def positional_encoding_2d_as(x: torch.Tensor, temperature: float = 1e4, scale: float = 2 * math.pi):
    d, h, w = x.shape[-3:]
    return positional_encoding_2d((d, h, w), temperature, scale, x.dtype, x.device).expand_as(x)

def attention(q, k, v, scale, mask=None):
    '''
    Args:
        q: *, N1, c
        k: *, N2, c
        v: *, N2, c
        mask: *, N1, N2
        scale:

    Returns:
        out: *, N1, c
    '''
    atten = q @ k.transpose(-2, -1) * scale
    if mask is not None:
        atten = atten.masked_fill(mask < 0, -1e9)
    atten = torch.softmax(atten, dim=-1)
    out = atten @ v

    return out