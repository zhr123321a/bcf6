"""Export/reload real BC6H DDS and row-major FP16 MLP parameters."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from .codec import pack_blocks, unpack_blocks, save_dds, load_dds, independent_decode, WEIGHTS
from .model import BCF6


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


@torch.no_grad()
def export_model(model,folder,identity=None):
    if model.mode!='quantized':raise ValueError('Export requires frozen quantized features')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    files=[];blocks=[]
    for i,pyramid in enumerate(model.pyramids):
        path=folder/f'features_{i}.dds'
        mips=[pack_blocks(*mip.integer_parameters(),signed=True) for mip in pyramid]
        receipt=save_dds(path,pyramid[0].width,pyramid[0].height,mips,True)
        files.append(dict(name=path.name,sha256=sha(path),**receipt));blocks.append(receipt['payload_bytes'])
    path=folder/'decoder_fp16.bin';layouts=[];offset=0;parts=[]
    for name,param in model.decoder.named_parameters():
        value=param.detach().cpu().numpy().astype('<f2')
        if not np.isfinite(value).all():raise ValueError('Nonfinite FP16 decoder')
        parts.append(value.tobytes());layouts.append(dict(name=name,shape=list(value.shape),offset_bytes=offset,bytes=value.nbytes));offset+=value.nbytes
    path.write_bytes(b''.join(parts));files.append(dict(name=path.name,sha256=sha(path),payload_bytes=offset,package_bytes=offset,header_bytes=0))
    payload=sum(f['payload_bytes'] for f in files)
    metadata=dict(format='bcf6_mode10_signed_v1',config=model.config(),network_layout=layouts,files=files,
                  identity=identity,feature_bytes_by_layer=blocks,decoder_bytes=offset,payload_bytes=payload,
                  payload_bpp=payload*8/(model.target_size[0]*model.target_size[1]),
                  byte_scope='payload_excluding_metadata_v1',normal_layout='XY_reconstruct_Z' if model.outputs==8 else 'RGB9',
                  feature_filter='bilinear+LOD; border; align_corners=False',
                  decoder_arithmetic='FP32 evaluation of saved FP16 weights',
                  hardware_shader_validation='pending')
    save_json(folder/'metadata.json',metadata)
    return metadata


@torch.no_grad()
def load_model(folder,device='cpu'):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    if meta['format']!='bcf6_mode10_signed_v1':raise ValueError('Unsupported deployment format')
    for row in meta['files']:
        if sha(folder/row['name'])!=row['sha256']:raise ValueError('Saved file identity mismatch')
        if (folder/row['name']).stat().st_size!=row['package_bytes']:raise ValueError('Saved size mismatch')
    model=BCF6(**meta['config']).to(device)
    for i,pyramid in enumerate(model.pyramids):
        signed,mips=load_dds(folder/f'features_{i}.dds')
        if not signed or len(mips)!=len(pyramid):raise ValueError('Saved mip format mismatch')
        for mip,(w,h,data) in zip(pyramid,mips):
            if (w,h)!=(mip.width,mip.height):raise ValueError('Saved mip shape mismatch')
            decoded=independent_decode(data,w,h,True)
            if not np.isfinite(decoded).all():raise ValueError('Nonfinite BC6H feature')
            mip.frozen_texture.copy_(torch.from_numpy(decoded).to(device))
            q,idx,p=unpack_blocks(data,True)
            mip.endpoints.copy_(torch.as_tensor(q,device=device))
            mip.indices.copy_(torch.as_tensor(WEIGHTS[idx]/64,device=device))
            mip.partitions.copy_(torch.as_tensor(p,device=device));mip.refresh_partition_table()
    weights=(folder/'decoder_fp16.bin').read_bytes()
    parameters=dict(model.decoder.named_parameters())
    consumed=0
    for layout in meta['network_layout']:
        p=parameters.pop(layout['name']);a=layout['offset_bytes'];b=a+layout['bytes']
        if a!=consumed or list(p.shape)!=layout['shape']:raise ValueError('Saved network layout mismatch')
        value=np.frombuffer(weights[a:b],dtype='<f2').astype(np.float32).reshape(layout['shape'])
        p.copy_(torch.from_numpy(value).to(device));consumed=b
    if parameters or consumed!=len(weights):raise ValueError('Incomplete network layout')
    return model,meta


def prediction_rgb9(prediction):
    if prediction.shape[-1]==9:return prediction
    xy=prediction[:,3:5]*2-1
    z=torch.sqrt((1-(xy*xy).sum(-1,keepdim=True)).clamp_min(0))*.5+.5
    return torch.cat([prediction[:,:5],z,prediction[:,5:]],-1)


@torch.no_grad()
def score_model(model,reference_rgb9,batch=65536):
    h,w,c=reference_rgb9.shape
    if c!=9:raise ValueError('RGB9 target required')
    device=next(model.parameters()).device
    sums=torch.zeros(9,dtype=torch.float64,device=device);sums8=torch.zeros(8,dtype=torch.float64,device=device)
    sel=[0,1,2,3,4,6,7,8]
    for start in range(0,w*h,batch):
        ids=torch.arange(start,min(start+batch,w*h),device=device)
        uv=torch.stack([(ids%w+.5)/w,(ids//w+.5)/h],-1)
        pred=prediction_rgb9(model(uv,0.))
        target=reference_rgb9.reshape(-1,9)[start:start+len(ids)].to(device)
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite reconstruction')
        err=(pred.double()-target.double()).square();sums+=err.sum(0);sums8+=err[:,sel].sum(0)
    mse=(sums/(w*h)).cpu().tolist();m8=float((sums8/(w*h)).mean());m9=sum(mse)/9
    def psnr(v):return -10*math.log10(v) if v>0 else None
    return dict(rgb9_mse=m9,rgb9_psnr=psnr(m9),trained8_mse=m8,trained8_psnr=psnr(m8),mse_per_channel=mse,
                zero_mse_psnr='infinite represented by null',peak=1,clamped=False,rgb8_rounded=False,mip=0,
                rgb9_normal_Z='derived_from_predicted_XY' if model.outputs==8 else 'learned',
                quality_scope='saved package independent BC6H decode; FP32 arithmetic with saved FP16 network')


@torch.no_grad()
def replay_check(cpu_model,other_model,batch=32768):
    w,h=cpu_model.target_size;device=next(other_model.parameters()).device
    maximum=0.;sse=0.;count=0
    for start in range(0,w*h,batch):
        ids=torch.arange(start,min(start+batch,w*h))
        uv=torch.stack([(ids%w+.5)/w,(ids//w+.5)/h],-1)
        a=prediction_rgb9(cpu_model(uv,0.))
        b=prediction_rgb9(other_model(uv.to(device),0.)).cpu()
        delta=a.double()-b.double()
        if not torch.isfinite(delta).all():raise ValueError('Nonfinite material replay')
        maximum=max(maximum,float(delta.abs().max()));sse+=float(delta.square().sum());count+=delta.numel()
    rms=math.sqrt(sse/count);strict=maximum<5e-5 and rms<1e-6;allowed=maximum<.05 and rms<.01
    return dict(max_abs=maximum,rms=rms,pixel_strict_pass=strict,strict_pass=strict,continuation_allowed=allowed,
                severity='pass' if strict else ('warning' if allowed else 'error'),native_warnings=[],
                strict_thresholds={'max_abs_exclusive':5e-5,'rms_exclusive':1e-6},
                gross_thresholds={'max_abs_inclusive':.05,'rms_inclusive':.01},
                measurement_policy_version='bcf6_original_material_replay_v1_20261026',
                scope='all mip0 pixels; original RGB9; unclamped; same independently decoded BC6H and FP16 weights',
                scalars=count,comparison_device=str(device))
