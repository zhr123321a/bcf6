"""Four independent mip pyramids, continuous BC6H parameters and a small MLP."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .codec import PARTITIONS, WEIGHTS, unquantize, pack_blocks, decode_parameters, blocks_to_image
from .storage import PROFILES, mip_shapes


def halfcode_to_float(code):
    """Continuous piecewise-linear extension of signed finite half bit patterns."""
    a=code.abs().clamp(max=31743)
    exponent=torch.floor(a/1024)
    normal=torch.sign(code)*torch.pow(2.,exponent-15)*(1+a/1024-exponent)
    return torch.where(a<1024,code*(2.**-24),normal)


def float_to_halfcode(value):
    a=value.abs().clamp(min=2.**-24,max=65504.)
    exponent=torch.floor(torch.log2(a)).clamp(-14,15)
    normal=(exponent+15)*1024+(a/torch.pow(2.,exponent)-1)*1024
    code=torch.where(a<2.**-14,a*(2.**24),normal)
    return torch.where(value==0,torch.zeros_like(code),torch.sign(value)*code)


def torch_integer_palette(q):
    """Exact signed Mode 10 palette, shape N x 2 x 8 x 3."""
    a=q.abs().to(torch.int32)
    u=torch.where(a==0,0,torch.where(a>=31,32767,((a<<15)+16384)>>5))
    u=torch.where(q<0,-u,u).reshape(-1,2,2,3)
    w=torch.as_tensor(WEIGHTS,device=q.device).view(1,1,8,1)
    v=(u[:,:,0,None]*(64-w)+u[:,:,1,None]*w+32)>>6
    code=(v.abs()*31)>>5
    bits=code | torch.where(v<0,32768,0)
    return bits.to(torch.int16).view(torch.float16).float()


@torch.no_grad()
def initialize_blocks(image, chunk_blocks=4096):
    """Mode 10 compressor: all 32 partitions, PCA lines in half-code domain.

    This encoder is a disclosed reproduction choice; the paper does not name
    its initialization encoder. Selection scores the exact signed BC6H palette.
    """
    h,w,_=image.shape
    x=image.reshape(h//4,4,w//4,4,3).permute(0,2,1,3,4).reshape(-1,16,3)
    table=torch.as_tensor(PARTITIONS & 1,device=x.device,dtype=torch.long)
    qlevels=torch.arange(32,device=x.device,dtype=torch.int32)
    ulut=torch.as_tensor(unquantize(np.arange(32,dtype=np.int32)),device=x.device)
    clut=((ulut*31)>>5).float()
    resultq=[];resulti=[];resultp=[]
    for start in range(0,len(x),chunk_blocks):
        values=x[start:start+chunk_blocks]
        code=float_to_halfcode(values)
        n=len(values);row=torch.arange(n,device=x.device)
        best=torch.full((n,),float('inf'),device=x.device)
        bq=torch.empty((n,4,3),device=x.device);bi=torch.empty((n,16),device=x.device);bp=torch.empty(n,dtype=torch.long,device=x.device)
        for p in range(32):
            endpoints=[]
            for subset in (0,1):
                sel=code[:,table[p]==subset]
                mean=sel.mean(1);centered=sel-mean[:,None]
                cov=centered.transpose(1,2)@centered
                axis=cov.diagonal(dim1=1,dim2=2).argmax(-1)
                direction=F.one_hot(axis,3).to(code.dtype)
                for _ in range(8):
                    v=(cov@direction[...,None])[...,0]
                    norm=v.norm(dim=-1,keepdim=True)
                    direction=torch.where(norm>1e-12,v/norm.clamp_min(1e-12),direction)
                proj=(centered*direction[:,None]).sum(-1)
                endpoints += [mean+proj.min(1).values[:,None]*direction,mean+proj.max(1).values[:,None]*direction]
            raw=torch.stack(endpoints,1)
            # Nearest actual endpoint code includes zero and the saturated edge.
            q=(raw.abs()[...,None]-clut).abs().argmin(-1).to(torch.int32)*torch.sign(raw).to(torch.int32)
            palette=torch_integer_palette(q)
            candidates=palette[:,table[p]]
            error=((candidates-values[:,:,None])**2).sum(-1)
            ix=error.argmin(-1);score=error.gather(2,ix[...,None])[...,0].sum(-1)
            improved=score<best
            best=torch.minimum(best,score)
            bq[improved]=q[improved].float();bi[improved]=torch.as_tensor(WEIGHTS,device=x.device)[ix[improved]].float()/64
            bp[improved]=p
        resultq.append(bq);resulti.append(bi);resultp.append(bp)
    return torch.cat(resultq),torch.cat(resulti),torch.cat(resultp)


class FeatureMip(nn.Module):
    def __init__(self,width,height,mode='free'):
        super().__init__();self.width=width;self.height=height;self.mode=mode
        n=(width//4)*(height//4)
        if mode=='free':
            self.pixels=nn.Parameter(torch.randn(height,width,3)*0.05)
        else:
            self.endpoints=nn.Parameter(torch.zeros(n,4,3))
            self.indices=nn.Parameter(torch.full((n,16),0.5))
            self.register_buffer('partitions',torch.zeros(n,dtype=torch.long))
            self.register_buffer('subset',torch.zeros(n,16,dtype=torch.long),persistent=False)
            self.register_buffer('unquant_lut',torch.as_tensor(unquantize(np.arange(-31,32,dtype=np.int32)),dtype=torch.float32),persistent=False)
            if mode=='quantized':
                self.register_buffer('frozen_texture',torch.zeros(height,width,3))

    def refresh_partition_table(self):
        if self.mode!='free':
            self.subset=torch.as_tensor(PARTITIONS & 1,device=self.partitions.device,dtype=torch.long)[self.partitions]

    def texture(self):
        if self.mode=='free':return self.pixels
        if self.mode=='quantized':return self.frozen_texture
        e=self.endpoints.clamp(-31,31)+31
        lo=e.floor().long();hi=(lo+1).clamp(max=62);f=e-lo
        u=self.unquant_lut[lo]*(1-f)+self.unquant_lut[hi]*f
        row=torch.arange(len(e),device=e.device)[:,None]
        a,b=u[row,2*self.subset],u[row,2*self.subset+1]
        t=self.indices.clamp(0,1)[...,None]
        code=((1-t)*a+t*b)*(31./32.)
        decoded=halfcode_to_float(code)
        return decoded.reshape(self.height//4,self.width//4,4,4,3).permute(0,2,1,3,4).reshape(self.height,self.width,3)

    @torch.no_grad()
    def initialize_from(self,image):
        e,i,p=initialize_blocks(image)
        self.endpoints.copy_(e);self.indices.copy_(i);self.partitions.copy_(p)
        self.refresh_partition_table()

    @torch.no_grad()
    def integer_parameters(self):
        q=self.endpoints.round().clamp(-31,31).cpu().numpy().astype(np.int32)
        weights=torch.as_tensor(WEIGHTS,device=self.indices.device).float()/64
        idx=(self.indices.clamp(0,1)[...,None]-weights).abs().argmin(-1).cpu().numpy().astype(np.int32)
        p=self.partitions.cpu().numpy().astype(np.int32)
        return q,idx,p

    @torch.no_grad()
    def freeze_quantized(self):
        q,idx,p=self.integer_parameters()
        data=blocks_to_image(decode_parameters(q,idx,p),self.width,self.height)
        self.mode='quantized'
        self.register_buffer('frozen_texture',torch.as_tensor(data,device=self.endpoints.device))
        self.endpoints.requires_grad_(False);self.indices.requires_grad_(False)

    @torch.no_grad()
    def constrain(self):
        if self.mode=='block':
            self.endpoints.clamp_(-31,31);self.indices.clamp_(0,1)


def sample_image(image,uv,mode='bilinear'):
    pixels=image.permute(2,0,1)[None]
    grid=(uv*2-1).reshape(1,1,-1,2)
    return F.grid_sample(pixels,grid,mode=mode,padding_mode='border',align_corners=False)[0,:,0].T


class BCF6(nn.Module):
    def __init__(self,profile='1K',outputs=8,target_size=(2048,2048),mode='free',sides=None):
        super().__init__();self.profile=profile;self.outputs=outputs;self.target_size=tuple(target_size);self.mode=mode
        self.sides=list(PROFILES[profile] if sides is None else sides)
        self.pyramids=nn.ModuleList([nn.ModuleList([FeatureMip(w,h,mode) for w,h in mip_shapes(n,n)]) for n in self.sides])
        self.decoder=nn.Sequential(nn.Linear(12,16),nn.ReLU(),nn.Linear(16,outputs))
        if mode=='quantized':
            for pyramid in self.pyramids:
                for mip in pyramid:
                    mip.endpoints.requires_grad_(False);mip.indices.requires_grad_(False)

    def features(self,uv,scale):
        result=[]
        for side,pyramid in zip(self.sides,self.pyramids):
            bias=math.log2(max(side/self.target_size[0],side/self.target_size[1]))
            lod=max(0.,min(float(scale)+bias,len(pyramid)-1))
            lo=int(lod);hi=min(lo+1,len(pyramid)-1);fraction=lod-lo
            value=sample_image(pyramid[lo].texture(),uv)
            if fraction>0:value=(1-fraction)*value+fraction*sample_image(pyramid[hi].texture(),uv)
            result.append(value)
        return torch.cat(result,-1)

    def forward(self,uv,scale):return self.decoder(self.features(uv,scale))

    @torch.no_grad()
    def to_blocks(self):
        model=BCF6(self.profile,self.outputs,self.target_size,'block',self.sides).to(next(self.parameters()).device)
        model.decoder.load_state_dict(self.decoder.state_dict())
        for oldp,newp in zip(self.pyramids,model.pyramids):
            for old,new in zip(oldp,newp):new.initialize_from(old.texture())
        return model

    @torch.no_grad()
    def freeze_quantized(self):
        for pyramid in self.pyramids:
            for mip in pyramid:mip.freeze_quantized()
        self.mode='quantized'

    @torch.no_grad()
    def constrain(self):
        for pyramid in self.pyramids:
            for mip in pyramid:mip.constrain()

    def config(self):
        return dict(profile=self.profile,outputs=self.outputs,target_size=list(self.target_size),mode=self.mode,sides=self.sides)

    @classmethod
    def from_state(cls,config,state,device='cpu'):
        model=cls(**config).to(device);model.load_state_dict(state)
        for pyramid in model.pyramids:
            for mip in pyramid:mip.refresh_partition_table()
        return model
