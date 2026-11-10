"""Exercise the real CAS emitters with interleaved writers sharing packed words."""
from pathlib import Path
import os, shutil, subprocess, tempfile, unittest
ROOT = Path(__file__).resolve().parents[1]
TRANS = ROOT / 'nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/Translation'

def method(source, name):
    start = source.index('        private static Operand ' + name + '(')
    opening = source.index('{', start)
    depth = 1
    i = opening + 1
    while depth:
        depth += (source[i] == '{') - (source[i] == '}')
        i += 1
    return source[start:i]

MOCK = r'''
using System;
using System.Collections.Generic;
enum StorageKind { GlobalMemory, StorageBuffer, SharedMemory }
sealed class Operand { uint value;public Func<uint> Read;public uint Value {get=>Read==null?value:Read();set=>this.value=value;} public int Target; public Operand(uint v=0) {Value=v;} }
sealed class Memory { public uint Word; public int Seeds, Loads, Cas, Failed; }
sealed class EmitterContext {
 public readonly Memory Mem; public readonly List<Action> Ops=new(); public int Pc;
 public EmitterContext(Memory m) {Mem=m;}
 Operand Add(Func<uint> f) {var v=new Operand();Ops.Add(()=>v.Value=f());return v;}
 public void MarkLabel(Operand label) {label.Target=Ops.Count;}
 public void Copy(Operand dst,Operand src) {Ops.Add(()=>dst.Value=src.Value);}
 public void BranchIfFalse(Operand label,Operand test) {Ops.Add(()=>{if(test.Value==0)Pc=label.Target;});}
 public void Branch(Operand label) {Ops.Add(()=>Pc=label.Target);}
 // The production AST emitter may inline comparisons into their consuming branch.
 public Operand ICompareNotEqual(Operand a,Operand b)=>new(){Read=()=>a.Value!=b.Value?1u:0u};
 public Operand IMinimumS32(Operand a,Operand b)=>Add(()=>unchecked((uint)Math.Min((int)a.Value,(int)b.Value)));
 public Operand IMaximumS32(Operand a,Operand b)=>Add(()=>unchecked((uint)Math.Max((int)a.Value,(int)b.Value)));
 public Operand BitfieldInsert(Operand old,Operand value,Operand offset,Operand bits)=>Add(()=>{
  uint mask=((1u<<(int)bits.Value)-1)<<(int)offset.Value;
  return (old.Value&~mask)|((value.Value<<(int)offset.Value)&mask);
 });
 public Operand AtomicOr(StorageKind k,params Operand[] args)=>Add(()=>{
  Mem.Seeds++;uint old=Mem.Word;Mem.Word|=args[^1].Value;return old;
 });
 public Operand AtomicOr(StorageKind k,int binding,Operand a,Operand b,Operand value)=>AtomicOr(k,new Operand((uint)binding),a,b,value);
 public Operand Load(StorageKind k,params Operand[] args)=>Add(()=>{Mem.Loads++;return Mem.Word;});
 public Operand Load(StorageKind k,int id,params Operand[] args)=>Load(k,args);
 public Operand AtomicCompareAndSwap(StorageKind k,params Operand[] args)=>Add(()=>{
  Mem.Cas++;uint old=Mem.Word;
  if(old==args[^2].Value)Mem.Word=args[^1].Value;else Mem.Failed++;
  return old;
 });
 public Operand AtomicCompareAndSwap(StorageKind k,int binding,params Operand[] args)=>AtomicCompareAndSwap(k,args);
 public bool Step() {if(Pc==Ops.Count)return false;if(Pc>Ops.Count)throw new Exception("bad jump");Ops[Pc++]();return true;}
}
static class Test {
 static Operand Const(int x)=>new(unchecked((uint)x));
 static Operand Label()=>new();
 static void Check(bool v,string text) {if(!v)throw new Exception(text);}
'''
CHECKS = r'''
 static void Schedule(List<EmitterContext> c,Random random,bool lockstep) {
  for(int steps=0;steps<100000;steps++) {
   bool active=false;
   if(lockstep) {foreach(var x in c)active|=x.Step();}
   else {foreach(var x in c)active|=x.Pc<x.Ops.Count;if(active)c[random.Next(c.Count)].Step();}
   if(!active)return;
  }
  throw new Exception("CAS loop failed to terminate");
 }
 static void Main() {
  var random=new Random(72219);
  int checks=0;
  for(int kind=0;kind<3;kind++) foreach(int bits in new[]{8,16}) for(int trial=0;trial<80;trial++) {
   var mem=new Memory{Word=0xa5b6c7d8};var contexts=new List<EmitterContext>();uint expected=mem.Word;
   int lanes=32/bits;
   for(int lane=0;lane<lanes;lane++) {
    var c=new EmitterContext(mem);contexts.Add(c);uint val=(uint)random.Next(1<<bits);
    int shift=bits*lane;uint mask=((1u<<bits)-1)<<shift;expected=(expected&~mask)|(val<<shift);
    if(kind==0)GenerateAtomicCasLoop(c,Const(3),5,x=>c.BitfieldInsert(x,new Operand(val),Const(shift),Const(bits)));
    else if(kind==1)GenerateSharedAtomicCasLoop(c,Const(3),5,x=>c.BitfieldInsert(x,new Operand(val),Const(shift),Const(bits)));
    else GenerateRawLoop(c,Const(0),Const(1),new Operand(val),Const(shift),bits);
   }
   Schedule(contexts,random,trial==0);
   Check(mem.Word==expected,$"packed write lost data kind={kind} bits={bits}");
   Check(mem.Loads==0 && mem.Seeds==lanes,"must seed atomically exactly once per invocation");
   if(trial==0)Check(mem.Failed>0,"lockstep must exercise a failed CAS");
   checks++;
  }
  foreach(bool max in new[]{false,true}) {
   var mem=new Memory{Word=unchecked((uint)-3)};
   var c=new EmitterContext(mem);
   Operand result=GenerateAtomicCasLoop(c,Const(0),1,x=>max?c.IMaximumS32(x,Const(9)):c.IMinimumS32(x,Const(-9)));
   Schedule(new(){c},random,true);
   Check((int)result.Value==-3,"return successful original value");
   Check((int)mem.Word==(max?9:-9),"signed update");
  }
  foreach(bool max in new[]{false,true}) {
   var mem=new Memory{Word=unchecked((uint)-3)};var cs=new List<EmitterContext>();var result=new List<Operand>();
   foreach(int n in new[]{9,12}) {
    var c=new EmitterContext(mem);cs.Add(c);
    result.Add(GenerateAtomicCasLoop(c,Const(0),1,x=>max?c.IMaximumS32(x,Const(n)):c.IMinimumS32(x,Const(-n))));
   }
   Schedule(cs,random,true);
   Check(mem.Failed>0 && (int)result[0].Value==-3 && (int)result[1].Value==(max?9:-9),"contended signed return values");
   Check((int)mem.Word==(max?12:-12),"contended signed update");
  }
  Console.WriteLine($"PASS {checks} interleaved packed-word cases; retries, atomic-only seed, signed return value");
 }
}
'''

class CasLoops(unittest.TestCase):
    def test_actual_emitters(self):
        dotnet=os.environ.get('DOTNET') or shutil.which('dotnet')
        if not dotnet:self.skipTest('requires .NET 10; set DOTNET')
        global_src=(TRANS/'Optimizations/GlobalToStorage.cs').read_text(encoding='utf-8')
        shared_src=(TRANS/'HelperFunctionManager.cs').read_text(encoding='utf-8')
        emit=method(global_src,'GenerateAtomicCasLoop')+method(shared_src,'GenerateSharedAtomicCasLoop')
        start=global_src.index('                    Operand lblLoopHead = Label();')
        end=global_src.index('context.MarkLabel(lblDone);',start)+len('context.MarkLabel(lblDone);')
        emit+='\nstatic void GenerateRawLoop(EmitterContext context,Operand hostLow,Operand hostHigh,Operand value,Operand bitOffset,int bitSize) {\n'+global_src[start:end]+'\n}'
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cas-loops-',dir=parent) as tmp:
            p=Path(tmp)
            (p/'test.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>', encoding='utf-8')
            (p/'Program.cs').write_text(MOCK+emit+CHECKS, encoding='utf-8')
            r=subprocess.run([dotnet,'run','--project',str(p/'test.csproj'),'-c','Release','--verbosity','quiet'],capture_output=True,text=True,timeout=120)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('PASS',r.stdout)
            print(r.stdout.strip())

if __name__=='__main__':unittest.main()

