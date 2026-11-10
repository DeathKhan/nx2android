#!/usr/bin/env python3
"""Execute the real Ampere warp emitters with a per-lane IR evaluator.

Distinct matrix tags in every logical warp expose cross-warp gathers; the old
emitter fails at host lane 32. Requires .NET 10 (DOTNET may name the executable).
No copied implementation of the emitters is used.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
INST = ROOT / 'nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/Instructions'


def method(source, name):
    import re
    m = re.search(r'        (?:public|private) static [^\n]*\b' + name + r'\(', source)
    if not m:
        raise AssertionError(name)
    start = m.start()
    brace = source.find('{', start)
    arrow = source.find('=>', start, source.find('\n', start))
    if arrow >= 0:
        return source[start:source.find(';', arrow) + 1]
    depth = 1
    i = brace + 1
    while depth:
        depth += (source[i] == '{') - (source[i] == '}')
        i += 1
    return source[start:i]


MOCK = r'''
using System;
using System.Linq;
using System.Collections.Generic;
enum StorageKind { Input, SharedMemory }
enum IoVariable { SubgroupLaneId }
enum RegisterType { Gpr, Uniform }
static class RegisterConsts { public const int RegisterZeroIndex=255, UniformZeroIndex=63; }
static class IrConsts { public const int True=-1; }
sealed class Operand {
    public uint[] V;
    public Operand(uint[] v) { V=v; }
}
sealed class AmpereOp {
    public int Ra=1, Rd=2, Op;
    public bool Signed, Any=true, U64;
    public AmpereOp() {}
    public AmpereOp(AmpereOp x) { Ra=x.Ra;Rd=x.Rd;Op=x.Op;Signed=x.Signed;Any=x.Any;U64=x.U64; }
    public int Bits(int a,int b) => a==78 && b==81 ? Op : 0;
    public bool Bit(int bit) => bit==73 ? (U64||Signed) : bit==79 && Any;
}
sealed class Accessor {
    public int Size;
    public int QueryHostSubgroupSize()=>Size;
}
sealed class Translator {
    public Accessor GpuAccessor=new();
}
sealed class EmitterContext {
    public Translator TranslatorContext=new();
    public AmpereOp CurrOp=new();
    public uint[] Shared;
    public bool[] Active;
    public int RawLaneLoads;
    public int N;
    public EmitterContext(int n) {
        N=n; TranslatorContext.GpuAccessor.Size=n;
        Active=Enumerable.Repeat(true,n).ToArray();
    }
    Operand Map(Func<int,uint> f)=>new(Enumerable.Range(0,N).Select(f).ToArray());
    public Operand Load(StorageKind k,IoVariable v) { RawLaneLoads++; return Map(i=>(uint)i); }
    public Operand Load(StorageKind k,int id,Operand index)=>Map(i=>Shared[index.V[i]]);
    public Operand Shuffle(Operand a,Operand index)=>Map(i=>a.V[index.V[i]]);
    public Operand BitwiseAnd(Operand a,Operand b)=>Map(i=>a.V[i]&b.V[i]);
    public Operand BitwiseOr(Operand a,Operand b)=>Map(i=>a.V[i]|b.V[i]);
    public Operand BitwiseExclusiveOr(Operand a,Operand b)=>Map(i=>a.V[i]^b.V[i]);
    public Operand IAdd(Operand a,Operand b)=>Map(i=>unchecked(a.V[i]+b.V[i]));
    public Operand ShiftLeft(Operand a,Operand b)=>Map(i=>a.V[i]<<(int)b.V[i]);
    public Operand ShiftRightU32(Operand a,Operand b)=>Map(i=>a.V[i]>>(int)b.V[i]);
    public Operand BitfieldExtractU32(Operand a,Operand shift,Operand count)=>Map(i=>(a.V[i]>>(int)shift.V[i])&((1u<<(int)count.V[i])-1));
    public Operand ICompareEqual(Operand a,Operand b)=>Map(i=>a.V[i]==b.V[i]?uint.MaxValue:0);
    public Operand ICompareNotEqual(Operand a,Operand b)=>Map(i=>a.V[i]!=b.V[i]?uint.MaxValue:0);
    public Operand IMinimumS32(Operand a,Operand b)=>Map(i=>unchecked((uint)Math.Min((int)a.V[i],(int)b.V[i])));
    public Operand IMaximumS32(Operand a,Operand b)=>Map(i=>unchecked((uint)Math.Max((int)a.V[i],(int)b.V[i])));
    public Operand IMinimumU32(Operand a,Operand b)=>Map(i=>Math.Min(a.V[i],b.V[i]));
    public Operand IMaximumU32(Operand a,Operand b)=>Map(i=>Math.Max(a.V[i],b.V[i]));
    public Operand ConditionalSelect(Operand p,Operand a,Operand b)=>Map(i=>p.V[i]!=0?a.V[i]:b.V[i]);
    public Operand Ballot(Operand p,int word) {
        uint bits=0;
        for(int j=0;j<32;j++) if(word*32+j<N && Active[word*32+j] && p.V[word*32+j]!=0) bits|=1u<<j;
        return Map(i=>bits);
    }
    public void Copy(Operand dst,Operand src) { dst.V=(uint[])src.V.Clone(); }
}
static class Program {
    static int N;
    static Dictionary<(int,RegisterType),Operand> Regs=new();
    static Operand Const(int x)=>new(Enumerable.Repeat(unchecked((uint)x),N).ToArray());
    static Operand Register(int r,RegisterType t) {
        if(!Regs.TryGetValue((r,t),out var v)) Regs[(r,t)]=v=Const(0);
        return v;
    }
    static Exception AFail(EmitterContext c,string why)=>new Exception(why);
    static void ASetPred(EmitterContext c,int p,Operand v) {}
    static void Check(bool ok,string msg) { if(!ok) throw new Exception(msg); }
    static uint Tag(int warp,int row,int col)=>(uint)(1+warp*1000+row*10+col);
    static uint Pair(uint a,uint b)=>a|(b<<16);
    static Operand Values(Func<int,uint> f)=>new(Enumerable.Range(0,N).Select(f).ToArray());
'''
CHECKS = r'''
    static void Main() {
        foreach(int width in new[]{32,64,128}) {
            N=width; Regs.Clear();
            var c=new EmitterContext(N);
            // Every possible lane source stays inside the caller's logical warp.
            var tagged=Values(i=>(uint)(i+500));
            for(int src=0;src<32;src++) {
                var got=AWarpShuffle(c,tagged,Const(src));
                for(int i=0;i<N;i++) Check(got.V[i]==500+(i&~31)+src,$"shuffle {N}/{i}/{src}");
            }
            if(N==32) Check(c.RawLaneLoads==0,"32-lane fast path changed");

            // MOVM's output matrix must be exactly each input matrix's transpose.
            c.Copy(Register(1,RegisterType.Gpr),Values(i=>Pair(Tag(i/32,(i%32)/4,2*(i%4)),Tag(i/32,(i%32)/4,2*(i%4)+1))));
            AMovm(c);
            for(int i=0;i<N;i++) {
                uint want=Pair(Tag(i/32,2*(i%4),(i%32)/4),Tag(i/32,2*(i%4)+1,(i%32)/4));
                Check(Register(2,RegisterType.Gpr).V[i]==want,$"MOVM {N}/{i}");
            }

            // Four independent 8x8 matrices per warp in shared memory; both LDSM forms.
            c.Shared=new uint[(N/32)*128];
            for(int w=0;w<N/32;w++) for(int m=0;m<4;m++) for(int r=0;r<8;r++) for(int t=0;t<4;t++)
                c.Shared[w*128+m*32+r*4+t]=Pair(Tag(w,m*8+r,t*2),Tag(w,m*8+r,t*2+1));
            var addr=Values(i=>(uint)((i/32)*512+(i%32)*16));
            var g=Values(i=>(uint)((i%32)/4));var tval=Values(i=>(uint)(i%4));
            var translator=c.TranslatorContext; c.TranslatorContext=null;
            for(int mode=0;mode<2;mode++) for(int m=0;m<4;m++) {
                var got=LdsmLane(c,g,tval,addr,0,m,mode,N);
                for(int i=0;i<N;i++) {
                    int w=i/32,r=(i%32)/4,col=2*(i%4);
                    uint want=mode==0?Pair(Tag(w,m*8+r,col),Tag(w,m*8+r,col+1)):
                        Pair(Tag(w,m*8+col,r),Tag(w,m*8+col+1,r));
                    Check(got.V[i]==want,$"LDSM {N}/{mode}/{m}/{i}");
                }
            }

            c.TranslatorContext=translator;
            // Ballot/reduction/match use different active masks and data in every warp.
            for(int i=0;i<N;i++) c.Active[i]=(i%32)%(2+i/32)==0;
            var mask=EmitBallot(c,Const(-1));
            var data=Values(i=>(uint)((i/32)*100+(i%32)%5+1));
            c.Copy(Register(1,RegisterType.Gpr),data);
            c.CurrOp.Op=3; c.CurrOp.Rd=3; ARedux(c);
            c.CurrOp.Rd=4; AMatch(c);
            for(int i=0;i<N;i++) {
                uint bits=0,sum=0,same=0;
                for(int j=0;j<32;j++) {
                    int k=(i&~31)+j;
                    if(c.Active[k]) { bits|=1u<<j;sum+=data.V[k];if(data.V[k]==data.V[i])same|=1u<<j; }
                }
                Check(mask.V[i]==bits,$"ballot {N}/{i}");
                Check(Register(3,RegisterType.Uniform).V[i]==sum,$"REDUX {N}/{i}");
                Check(Register(4,RegisterType.Gpr).V[i]==same,$"MATCH {N}/{i}");
            }
        }
        Console.WriteLine("PASS: real Ampere emitter logical warps 32/64/128; MOVM/LDSM/REDUX/MATCH");
    }
}
'''


class AmpereWarp(unittest.TestCase):
    def test_actual_emitters(self):
        dotnet = os.environ.get('DOTNET') or shutil.which('dotnet')
        if not dotnet:
            self.skipTest('requires .NET 10; set DOTNET to its executable')
        src = (INST / 'InstEmitAmpereTensor.cs').read_text(encoding='utf-8')
        # Prevent any new Ampere operation from bypassing the tested warp helper.
        self.assertEqual(src.count('context.Shuffle('), 1)
        self.assertNotIn('context.Ballot(', src)
        names = ['ALaneId','AGpr','AWarpShuffle','LdsmLane','AMovm','ARedux','AMatch']
        emitted = '\n'.join(method(src,n) for n in names)
        emitted += method((INST/'InstEmitWarp.cs').read_text(encoding='utf-8'),'EmitBallot')
        emitted += method((INST/'InstEmitMove.cs').read_text(encoding='utf-8'),'EmitLoadSubgroupLaneId')
        out = ROOT / 'target/tests';out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ampere-warp-',dir=out) as tmp:
            p=Path(tmp)
            (p/'test.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>', encoding='utf-8')
            (p/'Program.cs').write_text(MOCK+emitted+CHECKS, encoding='utf-8')
            run=subprocess.run([dotnet,'run','--project',str(p/'test.csproj'),'-c','Release','--verbosity','quiet'],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS:',run.stdout)
            print(run.stdout.strip())


if __name__=='__main__':
    unittest.main()

