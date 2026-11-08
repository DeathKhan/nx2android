"""Test the actual image-load emitter's declared resource capability."""
from pathlib import Path
import importlib.util, os, shutil, subprocess, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('warp',ROOT/'tests/test_ampere_warp.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
MOCK=r'''
using System;
using System.Collections.Generic;
enum Capability { StorageImageReadWithoutFormat }
enum TextureFormat { Unknown, R32Float }
enum AggregateType { S32, FP32 }
enum ImageOperandsMask { MaskNone }
static class Formats { public static AggregateType GetComponentType(this TextureFormat f)=>AggregateType.FP32; }
readonly struct SamplerType {
 public int Dimensions {get;} readonly int flags;
 public static SamplerType Array=>new(0,1);
 public SamplerType(int d,int f=0) {Dimensions=d;flags=f;}
 public static int operator &(SamplerType a,SamplerType b)=>a.flags&b.flags;
}
sealed class SpvInstruction {}
class AstOperation {}
sealed class AstTextureOperation:AstOperation {
 public TextureFormat Format;public SamplerType Type;public int Index;
 public int GetTextureSetAndBinding()=>7;
 public int GetSource(int i)=>i;
 public AggregateType GetVectorType(AggregateType t)=>t;
}
sealed class Resource { public TextureFormat Format; }
sealed class Properties {public Dictionary<int,Resource> Images=new();}
sealed class ImageDeclaration {
 public SpvInstruction Image=new(),ImagePointerType=new(),ImageType=new();public bool IsIndexed;
}
sealed class OperationResult {public OperationResult(AggregateType t,SpvInstruction r){}}
sealed class CodeGenContext {
 public Properties Properties=new();public Dictionary<int,ImageDeclaration> Images=new();
 public HashSet<Capability> Caps=new();public int Reads,Chains,NonUniform,Sources;
 public void AddCapability(Capability c)=>Caps.Add(c);
 public SpvInstruction Get(AggregateType t,int source) {Sources++;return new();}
 public SpvInstruction AccessChain(SpvInstruction t,SpvInstruction image,SpvInstruction index) {Chains++;return new();}
 public SpvInstruction Load(SpvInstruction t,SpvInstruction p)=>new();
 public SpvInstruction TypeVector(SpvInstruction t,int count)=>new();
 public SpvInstruction TypeS32()=>new();
 public SpvInstruction CompositeConstruct(SpvInstruction t,SpvInstruction[] e)=>new();
 public SpvInstruction GetType(AggregateType t)=>new();
 public SpvInstruction ImageRead(SpvInstruction t,SpvInstruction image,SpvInstruction coords,ImageOperandsMask m) {
  Reads++;
  if(Properties.Images[7].Format==TextureFormat.Unknown&&!Caps.Contains(Capability.StorageImageReadWithoutFormat))
   throw new Exception("OpImageRead lacks capability for declared resource");
  return new();
 }
}
static class Test {
 static void MarkNonUniform(CodeGenContext c,SpvInstruction i) {c.NonUniform++;}
 static SpvInstruction GetSwizzledResult(CodeGenContext c,SpvInstruction t,AggregateType a,int index)=>t;
 static void Check(bool c,string m) {if(!c)throw new Exception(m);}
'''
CHECKS=r'''
 static void Main() {
  int cases=0;
  foreach(var declared in new[]{TextureFormat.Unknown,TextureFormat.R32Float})
  foreach(var operand in new[]{TextureFormat.Unknown,TextureFormat.R32Float})
  foreach(bool indexed in new[]{false,true})
  foreach(bool array in new[]{false,true})
  foreach(int dims in new[]{1,2,3}) {
   var c=new CodeGenContext();
   c.Properties.Images[7]=new(){Format=declared};c.Images[7]=new(){IsIndexed=indexed};
   GenerateImageLoad(c,new AstTextureOperation{Index=0,Format=operand,Type=new SamplerType(dims,array?1:0)});
   Check(c.Caps.Contains(Capability.StorageImageReadWithoutFormat)==(declared==TextureFormat.Unknown),"declared/refined format capability");
   Check(c.Reads==1&&c.Chains==(indexed?1:0)&&c.NonUniform==(indexed?2:0),"indexed/plain path");
   Check(c.Sources==dims+(array?1:0)+(indexed?1:0),"coordinate/index consumption");
   cases++;
  }
  Console.WriteLine($"PASS {cases} declared/refined format, indexed/plain, array image-load cases");
 }
}
'''
class ImageReadCapability(unittest.TestCase):
    def test_actual_image_load_emitter(self):
        dotnet=os.environ.get('DOTNET') or shutil.which('dotnet')
        if not dotnet:self.skipTest('requires .NET 10; set DOTNET')
        src=(ROOT/'nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/CodeGen/Spirv/Instructions.cs').read_text()
        actual=m.method(src,'GenerateImageLoad')
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='image-read-',dir=parent) as tmp:
            p=Path(tmp)
            (p/'test.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>')
            (p/'Program.cs').write_text(MOCK+actual+CHECKS)
            r=subprocess.run([dotnet,'run','--project',str(p/'test.csproj'),'-c','Release','--verbosity','quiet'],capture_output=True,text=True,timeout=120)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('PASS',r.stdout)
            print(r.stdout.strip())
if __name__=='__main__':unittest.main()

