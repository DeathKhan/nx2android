#!/usr/bin/env python3
"""Execute actual copy-local type inference for integer texture-array layers."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/StructuredIr/StructuredProgram.cs"
spec = importlib.util.spec_from_file_location("warp", ROOT / "tests/test_ampere_warp.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

MOCK = r"""
using System;using System.Collections.Generic;
enum AggregateType {S32,FP32,Bool}
enum OperandType {LocalVariable,Constant,Argument}
enum Instruction {Copy,ConditionalSelect,TextureSample,Add}
[Flags] enum SamplerType {Texture1D=1,Texture2D=2,Texture3D=3,Array=256,Shadow=512}
static class SamplerExt { extension(SamplerType t) { public int Dimensions=>(int)t&255; } }
interface INode {}
class Operand { public OperandType Type=OperandType.LocalVariable; public List<INode> UseOps=new(); }
class Operation:INode {
 public Instruction Inst; public Operand Dest; public Operand[] Src;
 public int SourcesCount=>Src.Length;
 public Operand GetSource(int n)=>Src[n];
 public Operation(Instruction i,Operand d,params Operand[] s){Inst=i;Dest=d;Src=s;foreach(var x in s)x.UseOps.Add(this);}
}
class TextureOperation:Operation {
 public SamplerType Type;public int Set=2,Binding=5,SamplerSet=-1,SamplerBinding=-1;
 public TextureOperation(SamplerType t,params Operand[] s):base(Instruction.TextureSample,new Operand(),s){Type=t;}
}
readonly record struct SetBindingPair(int SetIndex,int Binding);
class TextureDefinition {public int ArrayLength;public TextureDefinition(int n){ArrayLength=n;}}
class ShaderProperties {public Dictionary<SetBindingPair,TextureDefinition> Textures=new();}
static class OperandInfo {public static AggregateType GetVarType(OperandType t)=>AggregateType.S32;}
static class InstructionInfo {
 public static AggregateType GetSrcVarType(Instruction i,int n)=>i==Instruction.ConditionalSelect&&n==0?AggregateType.Bool:AggregateType.FP32;
}
"""

CHECKS = r"""
}
static class Test {
 static void Check(bool v,string why){if(!v)throw new Exception(why);}
 static Operand L()=>new();
 static int Main(){
  int cases=0;
  foreach(int dim in new[]{1,2,3})
  foreach(bool array in new[]{false,true})
  foreach(bool indexedTex in new[]{false,true})
  foreach(int sampler in new[]{-1,0,1})
  foreach(bool shadow in new[]{false,true})
  foreach(int extra in new[]{0,1,4,12}) {
   var props=new ShaderProperties();
   props.Textures[new(2,5)]=new(indexedTex?65536:1);
   if(sampler>=0)props.Textures[new(0,9)]=new(sampler==1?2048:1);
   int prefix=(indexedTex?1:0)+(sampler==1?1:0);
   int count=prefix+dim+(array?1:0)+extra;
   var src=new Operand[count];for(int i=0;i<count;i++)src[i]=L();
   var tex=new TextureOperation((SamplerType)dim|(array?SamplerType.Array:0)|(shadow?SamplerType.Shadow:0),src);
   if(sampler>=0){tex.SamplerSet=0;tex.SamplerBinding=9;}
   for(int i=0;i<count;i++){
    var got=Actual.GetSourceVarType(tex,i,props);
    var want=array&&i==prefix+dim?AggregateType.S32:AggregateType.FP32;
    Check(got==want,$"source slot {i}/{count} dim{dim} array{array} tex{indexedTex} sampler{sampler}");
    Check(Actual.GetVarTypeFromUses(src[i],props)==want,"direct use mismatch");
    cases++;
   }
   if(array){
    var original=L();var middle=L();var terminal=L();
    new Operation(Instruction.Copy,middle,original);
    new Operation(Instruction.Copy,terminal,middle);
    // Replace layer source with this copied SSA value; two uses retain integer inference.
    var ss=(Operand[])src.Clone();ss[prefix+dim]=terminal;
    var a=new TextureOperation(tex.Type,ss){SamplerSet=tex.SamplerSet,SamplerBinding=tex.SamplerBinding};
    var b=new TextureOperation(tex.Type,ss){SamplerSet=tex.SamplerSet,SamplerBinding=tex.SamplerBinding};
    Check(Actual.GetVarTypeFromUses(original,props)==AggregateType.S32,"copy-chain layer must remain integer");
    // Phi lowering produces copies to one local from either predecessor.
    var left=L();var right=L();var phi=L();
    new Operation(Instruction.Copy,phi,left);new Operation(Instruction.Copy,phi,right);
    var ps=(Operand[])src.Clone();ps[prefix+dim]=phi;
    new TextureOperation(tex.Type,ps){SamplerSet=tex.SamplerSet,SamplerBinding=tex.SamplerBinding};
    Check(Actual.GetVarTypeFromUses(left,props)==AggregateType.S32,"left phi copy");
    Check(Actual.GetVarTypeFromUses(right,props)==AggregateType.S32,"right phi copy");
    foreach(int value in new[]{0,1,2,-1,16777216,int.MinValue,int.MaxValue}){
     // Integer storage preserves all guest bits; conversion occurs only at the sampler.
     int stored=value;float coordinate=(float)stored;
     Check(coordinate==(float)value,"integer-to-float conversion changed");
    }
    cases+=3;
   }
  }
  var empty=new ShaderProperties();var unused=L();
  Check(Actual.GetVarTypeFromUses(unused,empty)==AggregateType.S32,"unused fallback");
  var p=L();new Operation(Instruction.ConditionalSelect,L(),p,L(),L());
  Check(Actual.GetVarTypeFromUses(p,empty)==AggregateType.Bool,"predicate type changed");
  Console.WriteLine($"PASS {cases} texture-source positions/copy chains/phi copies; nonlayer types unchanged");
  return 0;
 }
}
"""

class TextureLayerTypes(unittest.TestCase):
    def test_actual_type_inference(self):
        dotnet=os.environ.get("DOTNET") or shutil.which("dotnet")
        if not dotnet:self.skipTest("requires .NET 10; set DOTNET")
        source=SOURCE.read_text()
        actual="\n".join(helper.method(source,n) for n in ("GetVarTypeFromUses","GetSourceVarType"))
        actual=actual.replace("private static","public static")
        code=MOCK+"\nstatic class Actual {\n"+actual+CHECKS
        parent=ROOT/"target/tests";parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="texture-layer-types-",dir=parent) as folder:
            p=Path(folder);project=p/"test.csproj"
            project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>')
            def run(text):
                (p/"Program.cs").write_text(text)
                return subprocess.run([dotnet,"run","--project",str(project),"-c","Release","--verbosity","quiet"],capture_output=True,text=True,timeout=120)
            r=run(code)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn("PASS",r.stdout);print(r.stdout.strip())
            old=code.replace("return GetSourceVarType(operation, index, properties);","return InstructionInfo.GetSrcVarType(operation.Inst, index);")
            self.assertNotEqual(code,old)
            bad=run(old);self.assertNotEqual(bad.returncode,0)
            self.assertIn("direct use mismatch",bad.stderr)

if __name__=="__main__":unittest.main()
