"""Small, stand-alone, spatially conditioned student generators.

No diffusion base model is loaded at inference.
Input contract: RGB(3), depth(1), normal(3), semantic one-hot(7), line(1).
"""
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS=15

class ConvNormAct(nn.Module):
    def __init__(self, ci,co,k=3,stride=1):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(ci,co,k,stride,k//2),nn.GroupNorm(min(8,co),co),nn.SiLU())
    def forward(self,x):return self.net(x)

class SemanticResBlock(nn.Module):
    """Very small SPADE-like affine conditioning, fixed categories not prompt tokens."""
    def __init__(self,dim,categories=7):
        super().__init__()
        self.norm1=nn.GroupNorm(min(8,dim),dim,affine=False)
        self.norm2=nn.GroupNorm(min(8,dim),dim,affine=False)
        self.mod=nn.Sequential(nn.Conv2d(categories,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,dim*2,3,padding=1))
        self.a=nn.Conv2d(dim,dim,3,padding=1)
        self.b=nn.Conv2d(dim,dim,3,padding=1)
    def forward(self,x,sem):
        cond=F.interpolate(sem,size=x.shape[-2:],mode='nearest')
        gamma,beta=self.mod(cond).chunk(2,1)
        h=F.silu(self.norm1(x)*(1+gamma)+beta)
        h=self.a(h);h=F.silu(self.norm2(h));h=self.b(h)
        return x+h*.4

class PooledAttention(nn.Module):
    """Bound quadratic attention to <=16*16 tokens at any output resolution."""
    def __init__(self,dim):
        super().__init__();self.norm=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)
    def forward(self,x):
        b,c,h,w=x.shape;hh=min(h,16);ww=min(w,16)
        pooled=F.adaptive_avg_pool2d(x,(hh,ww))
        tokens=pooled.flatten(2).transpose(1,2)
        q=self.norm(tokens);res,_=self.attn(q,q,q,need_weights=False)
        res=res.transpose(1,2).reshape(b,c,hh,ww)
        return x+.2*F.interpolate(res,size=(h,w),mode='bilinear',align_corners=False)

class Generator(nn.Module):
    def __init__(self,base=48,resblocks=3):
        super().__init__()
        # 48 -> 96 -> 192 -> 384 (fits a 10-30M engineering target)
        self.stem=ConvNormAct(CHANNELS,base,7)
        self.down1=ConvNormAct(base,base*2,stride=2)
        self.down2=ConvNormAct(base*2,base*4,stride=2)
        self.down3=ConvNormAct(base*4,base*8,stride=2)
        self.bottleneck=nn.ModuleList([SemanticResBlock(base*8) for _ in range(resblocks)])
        self.attention=PooledAttention(base*8)
        self.up2=ConvNormAct(base*8+base*4,base*4)
        self.up1=ConvNormAct(base*4+base*2,base*2)
        self.up0=ConvNormAct(base*2+base,base)
        self.final=nn.Conv2d(base,3,3,padding=1)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=CHANNELS:
            raise ValueError(f'expected BCHW with {CHANNELS} channels')
        sem=x[:,7:14]
        x0=self.stem(x);x1=self.down1(x0);x2=self.down2(x1);x3=self.down3(x2)
        for block in self.bottleneck:x3=block(x3,sem)
        x3=self.attention(x3)
        def up(src,skip,block):
            src=F.interpolate(src,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            return block(torch.cat([src,skip],1))
        y=up(x3,x2,self.up2);y=up(y,x1,self.up1);y=up(y,x0,self.up0)
        # Constrain style changes but permit transformation; prevents early all-gray collapse.
        return torch.clamp(x[:,:3]+.6*torch.tanh(self.final(y)),0,1)

def build_generator(route='A',base=None):
    if route not in {'A','B'}:raise ValueError('route must be A or B')
    if base is None:base=48 if route=='A' else 40
    return Generator(base=base,resblocks=3 if route=='A' else 2)

def parameter_count(model):return sum(p.numel() for p in model.parameters())

class PatchDiscriminator(nn.Module):
    def __init__(self,base=32):
        super().__init__();layers=[];ch=CHANNELS+3
        for co in [base,base*2,base*4,base*8]:
            layers.extend([nn.Conv2d(ch,co,4,2,1),nn.LeakyReLU(.2,inplace=True)])
            ch=co
        layers.append(nn.Conv2d(ch,1,3,padding=1));self.net=nn.Sequential(*layers)
    def forward(self,cond,rgb):return self.net(torch.cat([cond,rgb],1))
