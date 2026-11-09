#!/usr/bin/env python3
"""Compare actual Vulkan CPU upload/deswizzle output with same-run Mac GPU block bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=True)
    sources=[root/"probes/nvn_deswizzle_offline.c",root/"nxnative/dm/gfx/nvn/nvn_tex.c",root/"nxnative/dm/gfx/tex/nvn_deswizzle.c"]
    exe=a.out/"nvn_deswizzle_offline"
    build=["cc","-O1","-g","-fsanitize=address,undefined","-fno-omit-frame-pointer","-Wall","-Wextra","-Werror",*[str(s) for s in sources],"-o",str(exe)]
    subprocess.run(build,check=True)
    m=json.loads((a.capture/"same-run-material-manifest.json").read_text());results=[]
    for asset in m["assets"]:
        gpu_path=a.capture/Path(asset["gpu_raw"]).name
        guest_path=a.capture/Path(asset["guest_raw"]).name
        gpu=gpu_path.read_bytes();guest=guest_path.read_bytes()
        assert digest(gpu)==asset["gpu_raw_sha256"] and digest(guest)==asset["guest_sha256"]
        header,payload=gpu.split(b"\n",1);fields=list(map(int,header.split()))
        assert len(fields) in (4,5)
        w,h,fmt,pitch=fields[:4];layers=fields[4] if len(fields)==5 else 1
        assert (w,h,layers)==(asset["width"],asset["height"],asset["array_layers"])
        assert len(payload)==pitch*h*layers
        target=4 if layers>1 else 1
        # Explicit packaged tile byte and automatic layout coincide for these captures.
        pkg=(asset["block_depth_log2"]<<4)+asset["block_height_log2"]+1
        normal=None
        modes=[("normal",pkg,1),("automatic-layout",0,1)]
        if asset["name"]=="scalar-bc6h":modes.append(("negative-control-x1",pkg,0))
        for mode,block,gob2 in modes:
            prefix=a.out/(asset["name"]+"-"+mode)
            cmd=[str(exe),str(guest_path),str(prefix),str(target),asset["nvn_format"],str(asset["flags"]),"1",str(w),str(h),str(layers if layers>1 else 1),str(asset["levels"]),str(asset["stride"]),str(block),str(gob2)]
            run=subprocess.run(cmd,check=True,capture_output=True,text=True,env={**os.environ,"ASAN_OPTIONS":"detect_leaks=1"})
            layout=json.loads(run.stdout);sub=[]
            assert layout["size"]==asset["storage_bytes"]
            for r in layout["subresources"]:
                assert r["level"]==0 and r["slices"]==1
                rb,rows=r["row_bytes"],r["rows"];ly=r["layer"]
                actual=Path(r["path"]).read_bytes()
                expected=b"".join(payload[(ly*h+y)*pitch:(ly*h+y)*pitch+rb] for y in range(rows))
                differences=[i for i in range(0,len(actual),16) if actual[i:i+16]!=expected[i:i+16]]
                sub.append({**r,"mac_sha256":digest(expected),"vulkan_cpu_sha256":digest(actual),
                    "different_blocks":len(differences),"total_blocks":len(actual)//16,
                    "first_different_block_byte":differences[0] if differences else None})
            result={"asset":asset["name"],"mode":mode,"command":cmd,"layout":layout,"comparisons":sub,"all_equal":all(r["different_blocks"]==0 for r in sub)}
            results.append(result)
    report={"schema":"same-run-upload-deswizzle-comparison-v1",
        "capture_manifest":m,"sources":{str(s.relative_to(root)):digest(s.read_bytes()) for s in sources},
        "caller_sha256":digest((root/"nxnative/dm/android/gfx_vk_tex.c").read_bytes()),
        "compile":build,"results":results,
        "scope":"Actual nvn_tex_layout/nvn_deswizzle linked unchanged, same tight block rows as pkg_upload_ before vkCmdCopyBufferToImage; compares all level0 layers. It does not prove Android guest bytes, selected handles, native upload/copy or sampling.",
        "descriptor_assumptions":"2D_ARRAY uses NVN target4/depth=arrayLayers; 2D target1. Sample1 and stride0 from non-MS blocklinear captures. pkg_block_h=(BH|(BD<<4))+1 plus automatic0 both tested."}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+"\n")
    for r in results:print(r["asset"],r["mode"],"MATCH" if r["all_equal"] else "DIFFER",[(x["layer"],x["different_blocks"]) for x in r["comparisons"]])
    if not all(r["all_equal"] for r in results if r["mode"]!="negative-control-x1"):return 1
    if not all(not r["all_equal"] for r in results if r["mode"]=="negative-control-x1"):return 2
    return 0
if __name__=="__main__":raise SystemExit(main())

