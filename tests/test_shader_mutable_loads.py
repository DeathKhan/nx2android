#!/usr/bin/env python3
"""Run the actual AST optimizer against clobbered memory snapshots.

The old optimizer moves a local/storage read across a helper call or store.
Keep propagation and unused-read elimination as separate decisions.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/StructuredIr/AstOptimizer.cs"
spec = importlib.util.spec_from_file_location("warp", ROOT / "tests/test_ampere_warp.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

MOCK = r"""
using System;using System.Linq;using System.Collections.Generic;
enum Instruction {Copy,Load,ImageLoad,Call,Add,Store,AtomicAdd,AtomicAnd,AtomicCompareAndSwap,AtomicMaxS32,AtomicMaxU32,AtomicMinS32,AtomicMinU32,AtomicOr,AtomicSwap,AtomicXor,ImageAtomic,Mask=0xffff}
enum OperandType{LocalVariable,Constant}
enum StorageKind{None,Input,InputPerPatch,Output,OutputPerPatch,ConstantBuffer,StorageBuffer,LocalMemory,SharedMemory,SharedMemory8,SharedMemory16,GlobalMemory,GlobalMemoryS8,GlobalMemoryS16,GlobalMemoryU8,GlobalMemoryU16,GlobalMemoryVector}
enum TargetLanguage{Spirv}
interface IAstNode {}
class AstOperand:IAstNode {
 public OperandType Type;public int Value;public HashSet<IAstNode> Uses=new(),Defs=new();
 public AstOperand(int n,bool c=false){Value=n;Type=c?OperandType.Constant:OperandType.LocalVariable;}
}
class AstOperation:IAstNode {
 public Instruction Inst;public StorageKind StorageKind;IAstNode[] src;
 public int SourcesCount=>src.Length;
 public AstOperation(Instruction i,StorageKind s,params IAstNode[] v){Inst=i;StorageKind=s;src=new IAstNode[v.Length];for(int k=0;k<v.Length;k++)SetSource(k,v[k]);}
 public IAstNode GetSource(int i)=>src[i];
 public void SetSource(int i,IAstNode v){if(src[i] is AstOperand old)old.Uses.Remove(this);src[i]=v;if(v is AstOperand a)a.Uses.Add(this);}
}
class AstAssignment:IAstNode {
 public AstOperand Destination;IAstNode val;
 public IAstNode Source{get=>val;set{if(val is AstOperand o)o.Uses.Remove(this);val=value;if(value is AstOperand v)v.Uses.Add(this);}}
 public AstAssignment(AstOperand d,IAstNode s){Destination=d;d.Defs.Add(this);Source=s;}
}
class AstBlock:IAstNode {
 public List<IAstNode> Nodes=new();public IAstNode Condition;
 public void Remove(IAstNode n)=>Nodes.Remove(n);
}
class Fn{public AstBlock MainBlock=new();public HashSet<AstOperand> Locals=new();}
class StructuredProgramContext{public Fn CurrentFunction=new();public bool DebugMode;}
class AstBlockVisitor {
 public AstBlock Block;public AstBlockVisitor(AstBlock b){Block=b;}
 public IEnumerable<IAstNode> Visit()=>Block.Nodes.ToArray();
}
static class InstructionInfo{public static bool IsUnary(Instruction i)=>i==Instruction.Copy;}
"""
CHECKS = r"""
}
static class Test {
 static AstOperand C(int n)=>new(n,true);
 static bool Immutable(StorageKind k)=>k is StorageKind.ConstantBuffer or StorageKind.Input or StorageKind.InputPerPatch;
 static StructuredProgramContext Graph(StorageKind kind,bool call,bool nested=false,bool used=true,bool debug=false) {
  var c=new StructuredProgramContext{DebugMode=debug};var b=c.CurrentFunction.MainBlock;var v=new AstOperand(1);
  c.CurrentFunction.Locals.Add(v);
  b.Nodes.Add(new AstOperation(Instruction.Call,StorageKind.None,C(10)));
  IAstNode read=new AstOperation(Instruction.Load,kind,C(0));
  if(nested)read=new AstOperation(Instruction.Add,StorageKind.None,read,C(1));
  b.Nodes.Add(new AstAssignment(v,read));
  b.Nodes.Add(call?new AstOperation(Instruction.Call,StorageKind.None,C(20)):new AstOperation(Instruction.Store,kind,C(20)));
  b.Nodes.Add(new AstOperation(Instruction.Store,StorageKind.None,used?v:C(0)));
  return c;
 }
 static int Run(StructuredProgramContext c) {
  int memory=0,result=-1;var v=new Dictionary<AstOperand,int>();
  int Eval(IAstNode n) {
   if(n is AstOperand a)return a.Type==OperandType.Constant?a.Value:v[a];
   var op=(AstOperation)n;
   switch(op.Inst) {
    case Instruction.Call:memory=Eval(op.GetSource(0));return 0;
    case Instruction.Load:return Immutable(op.StorageKind)?7:memory;
    case Instruction.Copy:return Eval(op.GetSource(0));
    case Instruction.Add:return Eval(op.GetSource(0))+Eval(op.GetSource(1));
    case Instruction.Store:
     if(op.StorageKind==StorageKind.None)result=Eval(op.GetSource(0));else memory=Eval(op.GetSource(0));
     return 0;
    default:throw new Exception("unexpected operation");
   }
  }
  foreach(var n in c.CurrentFunction.MainBlock.Nodes)
   if(n is AstAssignment a)v[a.Destination]=Eval(a.Source);else Eval(n);
  return result;
 }
 static void Check(bool v,string why){if(!v)throw new Exception(why);}
 static void Main() {
  int cases=0;
  foreach(StorageKind kind in Enum.GetValues<StorageKind>().Where(k=>k!=StorageKind.None))
  foreach(bool call in new[]{false,true})
  foreach(bool nested in new[]{false,true}) {
   var c=Graph(kind,call,nested);int original=Run(c);
   Actual.Optimize(c,TargetLanguage.Spirv);int optimized=Run(c);
   Check(original==optimized,$"{kind} after {(call?"helper":"store")}: {original}->{optimized}");
   int count=c.CurrentFunction.MainBlock.Nodes.OfType<AstAssignment>().Count();
   Check(count==(Immutable(kind)?0:1),$"{kind}: snapshot/folding contract");
   cases++;
  }
  foreach(var kind in new[]{StorageKind.LocalMemory,StorageKind.StorageBuffer,StorageKind.SharedMemory,StorageKind.ConstantBuffer}) {
   var c=Graph(kind,true,used:false);Actual.Optimize(c,TargetLanguage.Spirv);
   Check(!c.CurrentFunction.MainBlock.Nodes.OfType<AstAssignment>().Any(),"unused read must remain removable");
   Check(Run(c)==0,"unused read removal changed output");cases++;
  }
  var dbg=Graph(StorageKind.LocalMemory,true,debug:true);Actual.Optimize(dbg,TargetLanguage.Spirv);
  Check(Run(dbg)==10,"debug mode changed");
  Console.WriteLine($"PASS {cases} mutable-load snapshots, nested reads, immutable folding and unused-load DCE cases");
 }
}
"""

class MutableLoads(unittest.TestCase):
    def test_actual_optimizer_keeps_reads_before_clobbers(self):
        dotnet = os.environ.get("DOTNET") or shutil.which("dotnet")
        if not dotnet:
            self.skipTest("requires .NET 10; set DOTNET")
        source = SOURCE.read_text(encoding='utf-8')
        names = ("Optimize", "HasSideEffects", "ReadsMutableMemory", "UsedByCall",
                 "IsWorthPropagating", "PropagateExpression")
        actual = chr(10).join(helper.method(source, name) for name in names)
        code = MOCK + chr(10) + "static class Actual {" + chr(10) + "static void RemoveEmptyBlocks(AstBlock b,TargetLanguage t){}" + chr(10) + actual + CHECKS
        parent = ROOT / "target/tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mutable-loads-", dir=parent) as folder:
            p = Path(folder)
            project = p / "test.csproj"
            project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>', encoding='utf-8')
            program = p / "Program.cs"
            def run(text):
                program.write_text(text, encoding='utf-8')
                return subprocess.run([dotnet, "run", "--project", str(project), "-c", "Release", "--verbosity", "quiet"],
                                      capture_output=True, text=True, timeout=120)
            result = run(code)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS", result.stdout)
            print(result.stdout.strip())
            # A semantic negative control: old propagation must fail the clobber test.
            old = code.replace(" && !ReadsMutableMemory(assignment.Source)", "")
            self.assertNotEqual(code, old)
            failed = run(old)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("after store", failed.stderr)

if __name__ == "__main__":
    unittest.main()
