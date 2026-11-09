#!/usr/bin/env python3
"""Prepare an identical compressed-image/fixed-coordinate workload for Metal and Vulkan."""
import argparse,hashlib,json,struct
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--asset",default="scalar-bc6h",choices=["scalar-bc6h","rgb-bc7"])
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();m=json.loads((a.capture/"same-run-material-manifest.json").read_text())
    asset=next(x for x in m["assets"] if x["name"]==a.asset)
    raw=(a.capture/Path(asset["gpu_raw"]).name).read_bytes()
    assert sha(raw)==asset["gpu_raw_sha256"]
    head,payload=raw.split(b"\n",1);w,h,fmt,pitch,layers=map(int,head.split())
    assert (w,h,layers)==(asset["width"],asset["height"],asset["array_layers"])
    assert fmt in (151,152) and len(payload)==pitch*h*layers
    rb=(w+3)//4*16;rows=(h+3)//4
    blocks=b"".join(payload[(ly*h+y)*pitch:(ly*h+y)*pitch+rb] for ly in range(layers) for y in range(rows))
    points=[];meta=[]
    def add(uv,layer,kind):
        tag=len(points);point=struct.unpack("<4f",struct.pack("<4f",uv[0],uv[1],layer,tag))
        points.append(point);meta.append({"index":tag,"uv":point[:2],"layer":layer,"kind":kind})
    for ly in range(layers):
        for y in range(9):
            for x in range(9):add((x/8,y/8),ly,"normalized_grid")
        xs=[0,1,3,4,w//4,w//2,w-4,w-1];ys=[0,1,3,4,h//4,h//2,h-4,h-1]
        for y in ys:
            for x in xs:add(((x+.5)/w,(y+.5)/h),ly,"texel_center")
        for x,y in [(3,3),(4,4),(w//2,h//2),(w-4,h-4)]:
            for fx,fy in [(0.,0.),(.25,.75),(.5,.5),(.75,.25)]:
                add(((x+.5+fx)/w,(y+.5+fy)/h),ly,"block_edge_filter")
        for u in [-.25,-1/w,0.,.5/w,1-.5/w,1.,1+1/w,1.25]:
            for v in [0.,.5,1.]:add((u,v),ly,"clamp_u")
        for v in [-.25,-1/h,0.,.5/h,1-.5/h,1.,1+1/h,1.25]:
            for u in [0.,.5,1.]:add((u,v),ly,"clamp_v")
    pts=b"".join(struct.pack("<4f",*x) for x in points);vk=143 if fmt==151 else 145
    header=struct.pack("<8s8I2Q8x",b"BCSAMP01",1,w,h,layers,vk,fmt,len(points),0,len(blocks),len(pts))
    assert len(header)==64
    data=header+blocks+pts;a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(data)
    manifest={"schema":"bc-sample-input-v1","file":str(a.out),"sha256":sha(data),"asset":asset,
        "tight_blocks_sha256":sha(blocks),"point_bytes_sha256":sha(pts),"point_count":len(points),
        "sampler":{"normalized":True,"min_filter":"linear","mag_filter":"linear","mipmap":"nearest","address_u":"clamp_to_edge","address_v":"clamp_to_edge","address_w":"clamp_to_edge","explicit_lod":0,"anisotropy":1,"comparison":False},
        "points":meta,"output":"raw float32 RGBA, point order exactly as listed; no image color conversion",
        "scope":"Standalone fixed-input hardware sampling only; no game bindings, interpolation, implicitLOD, or feedback."}
    Path(str(a.out)+".json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ["file","sha256","tight_blocks_sha256","point_count"]}))
if __name__=="__main__":main()

