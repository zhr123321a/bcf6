"""Exact theoretical storage for paper Table 1; no model is required."""
import argparse
import json

PROFILES={'0.5K':[512,256,128,64], '1K':[1024,512,256,128],
          '2K':[2048,1024,512,256], '2K++':[2048,2048,512,256]}


def mip_shapes(width,height):
    result=[]
    while True:
        result.append((width,height))
        if min(width,height)<=4:break
        width,height=max(1,width//2),max(1,height//2)
    return result


def estimate(profile,width=2048,height=2048,outputs=8):
    sides=PROFILES[profile]
    layers=[sum(((w+3)//4)*((h+3)//4)*16 for w,h in mip_shapes(n,n)) for n in sides]
    base=sum(n*n for n in sides)
    weights=2*(12*16+16+16*outputs+outputs)
    features=sum(layers);payload=features+weights
    return dict(profile=profile,width=width,height=height,output_channels=outputs,
                feature_bytes_by_layer=layers,base_mip_bytes=base,lower_mip_bytes=features-base,
                feature_bytes=features,decoder_fp16_bytes=weights,payload_bytes=payload,
                payload_bpp=8*payload/(width*height),feature_bpp=8*features/(width*height),
                payload_MiB=payload/2**20,dds_header_bytes=4*148,
                mip_policy='Table 1 count includes mip0 and ends at 4x4',
                byte_scope='payload_excluding_metadata_v1',measurement='theoretical exact layout count')


def main():
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=2048);p.add_argument('--height',type=int,default=2048)
    p.add_argument('--outputs',type=int,default=8);a=p.parse_args()
    if min(a.width,a.height,a.outputs)<=0:p.error('Positive dimensions required')
    print(json.dumps([estimate(k,a.width,a.height,a.outputs) for k in PROFILES],indent=2))


if __name__=='__main__':main()
