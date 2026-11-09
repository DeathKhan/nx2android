#!/usr/bin/env python3
"""Synthetic provenance and float-statistics regression for the actual analyzer."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np
import analyze_light_capture as a
ROOT=Path(__file__).resolve().parents[1]
class CaptureAnalysis(unittest.TestCase):
    def setUp(self):
        target=ROOT/"target/tests";target.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix="light-analysis-",dir=target)
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def image(self,name="pass_00.ppm",mismatch=False):
        rgb=np.array([[[0,2,np.nan],[np.inf,-.5,.5]]],dtype="<f4")
        preview=np.floor(np.where(np.isfinite(rgb),np.clip(rgb,0,1),1)*255+.5).astype(np.uint8)
        if mismatch:preview[:]=0
        (self.root/name).write_bytes(b"P6\n# comment\n2 1\n255\n"+preview.tobytes())
        (self.root/(name+".f32")).write_bytes(rgb.tobytes())
    def log(self,extra="",applied=True):
        text="[dm] lighting diagnostic: fragment abc mask 0x10 at support byte 1376\n"
        text+="[dm] pass dump: the next submission's passes from pass 0\n"
        if applied:text+="[dm] lighting diagnostic applied: fragment abc mask 0x10 present 20 frame 20 pass 3 draw 4 submission 90\n"
        text+="[dm] pass dump 00 is the submission's pass 3 (present 20)\n"
        text+="[dm] pass dump 00: 2x1 format 122 → /remote/pass_00.ppm\n"
        (self.root/"log.txt").write_text(text+extra)
    def buffer(self,name="fsbuf_0001_abc_s1_b3_draw4_before.bin",present=20):
        blob=b"\0"*16;(self.root/name).write_bytes(blob)
        meta={"fragment_key":"abc","shader_key":"abc","stage":1,"binding":3,"kind":4,"reg":1,
              "guest":"0x1000","native_offset":528,"descriptor_offset":512,"window_offset":16,
              "descriptor_range":64,"bytes":16,"written":16,"present":present,"frame":present,
              "pass":3,"draw":4,"fence":50,"phase":"before","result":0}
        (self.root/(name+".json")).write_text(json.dumps(meta))
        return f"[dm] graphics buffer landed 1: key abc dispatch 0 before binding 3 kind 4 reg 1 guest 0x1000 offset 528 range 64 bytes 16 written 16 fence 50 present {present} frame {present} result 0 -> /remote/{name}\n"
    def test_valid_mask_raw_values_and_stale_exclusion(self):
        self.image();self.image("pass_01.ppm");self.image("pass_02.ppm",True)
        extra="[dm] pass dump 02 is the submission's pass 4 (present 20)\n[dm] pass dump 02: 2x1 format 122 → /remote/pass_02.ppm\n"
        extra+=self.buffer();extra+=self.buffer("fsbuf_old.bin",19)
        self.log(extra)
        j=a.analyze(self.root,expect_mask=16)
        self.assertEqual(len(j["passes"]),1);self.assertEqual(len(j["buffers"]),1)
        self.assertEqual(len(j["excluded"]),3)
        st=j["passes"][0]["rgb_float"]
        self.assertEqual(st["finite_channels"],4)
        self.assertEqual(st["channels"][0]["positive_inf"],1)
        self.assertEqual(st["channels"][2]["nan"],1)
        self.assertEqual(st["channels"][1]["max"],2)
        self.assertEqual(st["channels"][1]["min"],-.5)
        self.assertEqual(j["passes"][0]["same_present_buffers"][0]["reg"],1)
        json.dumps(j,allow_nan=False)
    def test_pending_latest_cycle_does_not_reuse_old_outputs(self):
        self.image();self.log("[dm] pass dump: the next submission's passes from pass 0\n")
        j=a.analyze(self.root)
        self.assertFalse(j["passes"]);self.assertEqual(j["selected_cycle"],2)
    def test_unacknowledged_or_changed_control_rejected(self):
        self.image();self.log(applied=False)
        self.assertFalse(a.analyze(self.root,expect_mask=16)["passes"])
        self.log()
        path=self.root/"log.txt";text=path.read_text()
        text=text.replace("[dm] pass dump 00 is","[dm] lighting diagnostic: fragment 0 mask 0 at support byte 1376\n[dm] pass dump 00 is")
        path.write_text(text)
        self.assertFalse(a.analyze(self.root)["passes"])
    def test_ppm_does_not_skip_whitespace_valued_first_pixel(self):
        w,h,values=a.ppm(b"P6\n1 1\n255\n"+bytes([10,32,13]))
        self.assertEqual((w,h),(1,1));self.assertEqual(values.reshape(-1).tolist(),[10,32,13])
    def test_archive_duplicate_names_rejected_without_extraction(self):
        self.image();self.log()
        archive=self.root/"capture.tar"
        with tarfile.open(archive,"w") as t:
            for name in ("log.txt","pass_00.ppm","pass_00.ppm.f32"):
                data=(self.root/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
            data=(self.root/"pass_00.ppm").read_bytes();info=tarfile.TarInfo("../other/pass_00.ppm");info.size=len(data);t.addfile(info,io.BytesIO(data))
        j=a.analyze(archive)
        self.assertFalse(j["passes"]);self.assertIn("ambiguous duplicate",j["excluded"][0]["reason"])
        self.assertFalse((self.root.parent/"other").exists())
if __name__=="__main__":unittest.main()
