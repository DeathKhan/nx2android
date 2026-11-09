#!/usr/bin/env python3
"""Read-only correlation of bounded Android lighting dumps with their capture log.

No extraction from archives, no GPU/device access. Only files whose completion is
logged in the selected capture cycle are used. This is provenance checking, not
cryptographic proof that a device-side file was unmodified after the logged write.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import numpy as np

MAX_FILE = 256 << 20
ARM = re.compile(r"pass dump: the next submission's passes")
CONTROL = re.compile(r"lighting diagnostic: fragment ([0-9a-f]+) mask (0x[0-9a-f]+|[0-9]+) at support")
APPLIED = re.compile(r"lighting diagnostic applied: fragment ([0-9a-f]+) mask (0x[0-9a-f]+|[0-9]+) present (\d+) frame (\d+) pass (-?\d+) draw (\d+) submission (\d+)")
PASS = re.compile(r"pass dump (\d+) is the submission's pass (-?\d+) \(present (\d+)\)")
WRITE = re.compile(r"pass dump (\d+): (\d+)x(\d+) format (\d+) [→-]+>?\s*(\S+\.ppm)")
LANDED = re.compile(r"graphics buffer landed (\d+): key ([0-9a-f]+).*? written (\d+) fence (\d+) present (\d+) frame (\d+) result (-?\d+) -> (\S+)")
ERROR = re.compile(r"FATAL|VK_ERROR|endCommandBuffer.*(?:fail|error)|dropped submission|GPU timeline wait|query.*NOT_READY", re.I)
Q = [0, 1, 5, 25, 50, 75, 95, 99, 100]

def sha(data):
    return hashlib.sha256(data).hexdigest()

class Capture:
    def __init__(self, path):
        self.path = Path(path)
        self.tar = tarfile.open(self.path) if self.path.is_file() else None
        self.members = {}
        self.duplicates = set()
        items = self.tar.getmembers() if self.tar else sorted(self.path.rglob("*"))
        for item in items:
            if self.tar:
                if not item.isfile() or item.size > MAX_FILE:
                    continue
                name = Path(item.name).name
            else:
                if not item.is_file() or item.is_symlink() or item.stat().st_size > MAX_FILE:
                    continue
                name = item.name
            if name in self.members:
                self.duplicates.add(name)
            self.members[name] = item
    def read(self, name):
        if name in self.duplicates:
            raise ValueError("ambiguous duplicate basename: " + name)
        item = self.members[name]
        return self.tar.extractfile(item).read() if self.tar else item.read_bytes()
    def close(self):
        if self.tar:
            self.tar.close()

def ppm(data):
    tokens, pos = [], 0
    while len(tokens) < 4:
        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        if pos < len(data) and data[pos] == 35:
            end = data.find(b"\n", pos)
            if end < 0:
                raise ValueError("unfinished PPM comment")
            pos = end + 1
            continue
        end = pos
        while end < len(data) and data[end] not in b" \t\r\n":
            end += 1
        if end == pos:
            raise ValueError("short PPM header")
        tokens.append(data[pos:end])
        pos = end
    if tokens[0] != b"P6" or tokens[3] != b"255":
        raise ValueError("expected binary PPM8")
    w, h = map(int, tokens[1:3])
    if not 0 < w <= 16384 or not 0 < h <= 16384 or pos == len(data):
        raise ValueError("invalid PPM dimensions/header")
    pos += 2 if data[pos:pos+2] == b"\r\n" else 1
    if len(data) - pos != w*h*3:
        raise ValueError("PPM data length mismatch")
    return w, h, np.frombuffer(data, dtype=np.uint8, offset=pos).reshape(h, w, 3)

def stats(values):
    x = values.reshape(-1, 3)
    finite = np.isfinite(x)
    channels = []
    for i in range(3):
        v = x[:, i]
        f = v[np.isfinite(v)]
        channels.append({
            "channel": "RGB"[i], "finite": int(f.size),
            "nan": int(np.isnan(v).sum()), "positive_inf": int(np.isposinf(v).sum()),
            "negative_inf": int(np.isneginf(v).sum()),
            "min": float(f.min()) if f.size else None, "max": float(f.max()) if f.size else None,
            "mean": float(f.astype(np.float64).mean()) if f.size else None,
            "quantiles": dict(zip(map(str,Q), map(float,np.percentile(f.astype(np.float64),Q)))) if f.size else {},
            "below_zero": int((f < 0).sum()), "at_or_above_one": int((f >= 1).sum()),
            "above_one": int((f > 1).sum()),
        })
    return {"pixels": len(x), "finite_channels": int(finite.sum()),
            "finite_pixels": int(finite.all(axis=1).sum()), "channels": channels,
            "all_rgb_at_or_above_one_pixels": int((finite.all(axis=1) & (x >= 1).all(axis=1)).sum()),
            "note": ">=1 is display-preview clipping, not a physical HDR saturation limit."}

def parse_log(text):
    cycles = [{"id":0,"arm_line":None,"control":None,"writes":{}}]
    control = None
    applications, landings, errors, controls = [], {}, [], []
    pending = {}
    for line_no, line in enumerate(text.splitlines(),1):
        if m := CONTROL.search(line):
            control = {"line":line_no,"key":m[1],"mask":int(m[2],0)}
            controls.append(control)
        if m := APPLIED.search(line):
            applications.append(dict(zip(["key","mask","present","frame","pass","draw","submission"],
                                         [m[1],int(m[2],0)]+list(map(int,m.groups()[2:]))),line=line_no))
        if ARM.search(line):
            cycles.append({"id":len(cycles),"arm_line":line_no,"control":control,"writes":{}})
            pending = {}
        if m := PASS.search(line):
            pending[int(m[1])] = {"pass":int(m[2]),"present":int(m[3]),"metadata_line":line_no}
        if m := WRITE.search(line):
            index = int(m[1]); name = Path(m[5]).name
            if index in pending:
                cycles[-1]["writes"][name] = dict(pending[index],index=index,width=int(m[2]),
                    height=int(m[3]),format=int(m[4]),write_line=line_no)
        if m := LANDED.search(line):
            landings[Path(m[8]).name] = {"id":int(m[1]),"key":m[2],"written":int(m[3]),
                "fence":int(m[4]),"present":int(m[5]),"frame":int(m[6]),"result":int(m[7]),"line":line_no}
        if ERROR.search(line):
            errors.append({"line":line_no,"text":line[:500]})
    return cycles, applications, landings, errors, controls

def analyze(capture_path, log_path=None, cycle=None, expect_mask=None):
    cap=Capture(capture_path)
    try:
        rawlog=Path(log_path).read_bytes() if log_path else cap.read("log.txt")
        cycles, applications, landings, errors, controls = parse_log(rawlog.decode("utf-8","replace"))
        chosen=cycles[-1] if cycle is None else next(c for c in cycles if c["id"]==cycle)
        result={"schema":"android-lighting-capture-v1","capture":str(capture_path),
            "log":str(log_path) if log_path else "archive:log.txt","log_sha256":sha(rawlog),
            "selected_cycle":chosen["id"],"cycle_arm_line":chosen["arm_line"],"control_at_arm":chosen["control"],
            "applications":applications,"errors_in_supplied_log":errors,"passes":[],"buffers":[],"excluded":[],
            "warnings":[],"scope":"Each entry is one logged attachment pass. Cube layer identity is unknown unless separately logged; no whole-cube aggregation."}
        if chosen["arm_line"] is None:
            result["warnings"].append("Capture arm is outside supplied log; cycle provenance is incomplete.")
        if not chosen["writes"]:
            result["warnings"].append("Latest selected cycle has no completed PPM writes; older files are excluded.")
        complete_buffers=[]
        selected_presents={r["present"] for r in chosen["writes"].values()}
        for name in sorted(cap.members):
            if not re.fullmatch(r"fsbuf_.*\.bin\.json",name):
                continue
            try:
                meta=json.loads(cap.read(name)); binname=name[:-5]; blob=cap.read(binname)
                if meta["present"] not in selected_presents:
                    raise ValueError("buffer belongs to another capture presentation")
                landed=landings.get(binname)
                if not landed:
                    raise ValueError("no matching GPU buffer completion in supplied log")
                for k in ("fence","present","frame"):
                    if int(meta[k]) != landed[k]:
                        raise ValueError("buffer completion metadata mismatch: "+k)
                if meta["fragment_key"].lower()!=landed["key"].lower() or meta["result"] or landed["result"]:
                    raise ValueError("failed or mismatched buffer completion")
                if len(blob)!=meta["bytes"] or meta["written"]!=len(blob) or landed["written"]!=len(blob):
                    raise ValueError("short buffer output")
                off,window,span=map(int,(meta["descriptor_offset"],meta["window_offset"],meta["descriptor_range"]))
                if int(meta["native_offset"])!=off+window or window+len(blob)>span:
                    raise ValueError("buffer window outside actual descriptor")
                item={"file":binname,"sha256":sha(blob),**meta}
                complete_buffers.append(item)
            except (KeyError,ValueError,TypeError) as exc:
                result["excluded"].append({"file":name,"reason":str(exc)})
        result["buffers"]=complete_buffers
        for name in sorted(cap.members):
            if not re.fullmatch(r"pass_\d+\.ppm",name):
                continue
            record=chosen["writes"].get(name)
            if not record:
                result["excluded"].append({"file":name,"reason":"not logged as written in selected cycle"})
                continue
            try:
                data=cap.read(name);w,h,preview=ppm(data)
                if (w,h)!=(record["width"],record["height"]):
                    raise ValueError("PPM dimensions disagree with logged write")
                item={"file":name,**record,"ppm_sha256":sha(data),
                    "preview_white_pixels":int((preview==255).all(axis=2).sum()),
                    "preview_black_pixels":int((preview==0).all(axis=2).sum())}
                control=chosen["control"]
                item["mask"]=control["mask"] if control else None
                applied=[x for x in applications if control and x["key"]==control["key"] and
                         x["mask"]==control["mask"] and x["present"]<=record["present"] and
                         x["line"]<=record["write_line"] and x["line"]>=control["line"]]
                item["application_evidence"]=applied[-1] if applied else None
                for change in controls:
                    if not control or change["line"]<=chosen["arm_line"] or change["line"]>record["write_line"] or change["mask"]==control["mask"]:
                        continue
                    ack=[x for x in applications if x["line"]>=change["line"] and
                         x["line"]<=record["write_line"] and x["key"]==change["key"] and x["mask"]==change["mask"]]
                    if not ack or ack[0]["present"]<=record["present"]:
                        raise ValueError("mask changed during capture without an unambiguous later-presentation boundary")
                if control and control["mask"] and not applied:
                    raise ValueError("nonzero mask lacks actual selected-draw application evidence")
                if expect_mask is not None and item["mask"]!=expect_mask:
                    raise ValueError("mask does not match --expect-mask")
                side=name+".f32"
                if side in cap.members:
                    raw=cap.read(side)
                    if len(raw)!=w*h*3*4:
                        raise ValueError("sidecar is not complete RGB float32")
                    rgb=np.frombuffer(raw,dtype="<f4").reshape(h,w,3)
                    bounded=np.where(np.isfinite(rgb),np.clip(rgb,0,1),np.float32(1))
                    expected=np.floor(bounded*np.float32(255)+np.float32(.5)).astype(np.uint8)
                    difference=np.abs(expected.astype(np.int16)-preview.astype(np.int16))
                    if np.any(difference>1):
                        raise ValueError("float sidecar disagrees with PPM preview; stale/mismatched files")
                    item["float_sidecar_sha256"]=sha(raw)
                    item["preview_rounding_difference_channels"]=int((difference!=0).sum())
                    item["rgb_float"]=stats(rgb)
                else:
                    item["rgb_float"]=None
                    item["limitation"]="No complete RGB float sidecar; PPM alone clips HDR."
                item["same_present_buffers"]=[{"file":b["file"],"stage":b["stage"],"reg":b["reg"],
                    "binding":b["binding"],"kind":b["kind"],"shader_key":b["shader_key"],
                    "draw":b["draw"],"pass":b["pass"],"fence":b["fence"],"guest":b["guest"],
                    "window_offset":b["window_offset"],"bytes":b["bytes"]}
                    for b in complete_buffers if b["present"]==record["present"]]
                item["buffer_correlation_note"]="Same presentation only: buffer pass is recorded before diagnostic STORE/copy/LOAD; do not infer exact face or force equal pass numbers."
                result["passes"].append(item)
            except (KeyError,ValueError,TypeError) as exc:
                result["excluded"].append({"file":name,"reason":str(exc)})
        return result
    finally:
        cap.close()

def compact(result):
    rows=[f"cycle {result['selected_cycle']}: {len(result['passes'])} accepted attachment dumps, "
          f"{len(result['buffers'])} verified buffers, {len(result['excluded'])} excluded files"]
    for present in sorted({p["present"] for p in result["passes"]}):
        group=[p for p in result["passes"] if p["present"]==present]
        floats=[p for p in group if p["rgb_float"]]
        maxima=[max((p["rgb_float"]["channels"][c]["max"] for p in floats
                     if p["rgb_float"]["channels"][c]["max"] is not None),default=None) for c in range(3)]
        rows.append(f"P{present}: {len(group)} passes, maxRGB={maxima}, masks={sorted({p['mask'] for p in group},key=str)}")
    ranked=sorted([p for p in result["passes"] if p["rgb_float"] and p["rgb_float"]["channels"][0]["mean"] is not None],
                  key=lambda p:p["rgb_float"]["channels"][0]["mean"],reverse=True)[:8]
    for p in ranked:
        stat=p["rgb_float"];channels=stat["channels"]
        mean="/".join(f"{c['mean']:.5g}" if c["mean"] is not None else "none" for c in channels)
        mx="/".join(f"{c['max']:.5g}" if c["max"] is not None else "none" for c in channels)
        rows.append(f"{p['file']} P{p['present']} pass{p['pass']} {p['width']}x{p['height']} fmt{p['format']} mask={p['mask']}: "
                    f"meanRGB={mean} maxRGB={mx} finite={stat['finite_channels']}/{stat['pixels']*3}")
    rows += ["WARNING: "+w for w in result["warnings"]]
    return "\n".join(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture",type=Path,required=True,help="capture directory or tar/tar.gz; files are never extracted")
    p.add_argument("--log",type=Path,help="complete same-run log, otherwise log.txt from capture")
    p.add_argument("--cycle",type=int,help="explicit arm-cycle index; default is latest arm, even if incomplete")
    p.add_argument("--expect-mask",type=lambda x:int(x,0))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.capture,a.log,a.cycle,a.expect_mask)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    text=compact(result);a.out.with_suffix(".txt").write_text(text+"\n");print(text)
if __name__=="__main__":
    main()
