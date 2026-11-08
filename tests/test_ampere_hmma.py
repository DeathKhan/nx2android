"""Run the actual shared and shuffle HMMA emitters through a SIMD IR evaluator.

Checks arithmetic equivalence and every packed operand transport, including NaN
payloads, infinities and signed zero, in 32/64/128-wide hardware subgroups.
"""
from pathlib import Path
import importlib.util,subprocess,os,shutil,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('warp',ROOT/'tests/test_ampere_warp.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
src=(ROOT/'nxnative/dm/third_party/ryujinx/src/Ryujinx.Graphics.Shader/Translation/HelperFunctionManager.cs').read_text()
old=m.method(src,'GenerateHmma16816F16')
new=m.method(src,'GenerateHmma16816F16Shuffle')
# Add float-bit operations and shared-memory storage to the already tested SIMD mock.
mock=m.MOCK
mock=mock.replace('static class IrConsts', '''sealed class Function { public Function(object b,string n,bool r,int i,int o){} }
sealed class ControlFlowGraph { public object Blocks=>null;public static ControlFlowGraph Create(object x)=>new(); }
static class TensorPass { public const int HmmaInArguments=17; }
static class HmmaState { public static int N;public static Operand[] Args;public static uint[] Shared;public static Operand Result;public static List<uint[]> Shuffles=new(); }
static class IrConsts''')
mock=mock.replace('public Operand Shuffle(Operand a,Operand index)=>Map(i=>a.V[index.V[i]]);', 'public Operand Shuffle(Operand a,Operand index) { var value=Map(i=>a.V[index.V[i]]);HmmaState.Shuffles.Add((uint[])value.V.Clone());return value; }')
mock=mock.replace('    public EmitterContext(int n) {','''    public EmitterContext():this(HmmaState.N) { Shared=HmmaState.Shared; }
    public void Store(StorageKind k,int id,Operand addr,Operand value) { for(int i=0;i<N;i++)Shared[addr.V[i]]=value.V[i]; }
    public void SubgroupBarrier() {}
    public object GetOperations()=>null;
    public void Return(Operand v) { HmmaState.Result=v; }
    static float F(uint x)=>BitConverter.UInt32BitsToSingle(x);
    static uint U(float x)=>BitConverter.SingleToUInt32Bits(x);
    static float HalfF(uint x)=>(float)BitConverter.UInt16BitsToHalf((ushort)x);
    public Operand UnpackHalf2x16Low(Operand a)=>Map(i=>U(HalfF(a.V[i])));
    public Operand UnpackHalf2x16High(Operand a)=>Map(i=>U(HalfF(a.V[i]>>16)));
    public Operand PackHalf2x16(Operand a,Operand b)=>Map(i=>(uint)BitConverter.HalfToUInt16Bits((Half)F(a.V[i]))|((uint)BitConverter.HalfToUInt16Bits((Half)F(b.V[i]))<<16));
    public Operand FPFusedMultiplyAdd(Operand a,Operand b,Operand c)=>Map(i=>U(MathF.FusedMultiplyAdd(F(a.V[i]),F(b.V[i]),F(c.V[i]))));
    public EmitterContext(int n) {''')
mock=mock.replace('    static Operand Const(int x)', '    static Operand Argument(int i)=>HmmaState.Args[i];\n    static Operand Const(int x)')
checks=r'''
    static void Main() {
        Environment.SetEnvironmentVariable("DMSH_HMMA_SHUFFLE",null);
        var random=new Random(3187);
        uint finiteHalf() { uint x=(uint)random.Next(65536);return (x&0x7c00)==0x7c00?x&~0x7c00u:x; }
        int checkedPairs=0;
        foreach(int width in new[]{32,64,128}) for(int trial=0;trial<80;trial++) {
            N=width;HmmaState.N=N;
            HmmaState.Args=new Operand[18];
            HmmaState.Args[0]=Values(i=>(uint)(i%32));HmmaState.Args[1]=Values(i=>(uint)((i/32)*256));
            for(int k=2;k<18;k++) HmmaState.Args[k]=Values(i=>finiteHalf()|(finiteHalf()<<16));
            // Include all-zero and signed-zero rows, and finite extremes among randomized inputs.
            if(trial==0) for(int k=2;k<10;k++)HmmaState.Args[k]=Values(i=>i%2==0?0u:0x80008000u);
            if(trial==1) for(int k=2;k<10;k++)HmmaState.Args[k]=Values(i=>i%2==0?0x7bff7bffu:0xfbfffbffu);
            if(trial==2) for(int k=2;k<10;k++) {
                uint[] edge={0x7c00,0xfc00,0x7e01,0xfe7f,0x7c01,0xfc01,0,0x8000};
                HmmaState.Args[k]=Values(i=>edge[(i+k)%8]|(edge[(i+k+3)%8]<<16));
            }
            HmmaState.Shared=new uint[(N/32)*256]; HmmaState.Shuffles.Clear();
            GenerateHmma16816F16(0);
            var d0=(uint[])HmmaState.Result.V.Clone();var d1=(uint[])HmmaState.Args[17].V.Clone();
            GenerateHmma16816F16Shuffle();
            Check(HmmaState.Shuffles.Count==32,"expected 16 A + 16 B gathers");
            for(int q=0;q<2;q++) for(int kw=0;kw<8;kw++) for(int i=0;i<N;i++) {
                int block=(i/32)*256,g=(i%32)/4,t=i%4;
                Check(HmmaState.Shuffles[kw*4+q][i]==HmmaState.Shared[block+(g+8*q)*8+kw],"A packed transport");
                Check(HmmaState.Shuffles[kw*4+2+q][i]==HmmaState.Shared[block+128+(2*t+q)*8+kw],"B packed transport");
            }
            for(int i=0;i<N;i++) {
                Check(d0[i]==HmmaState.Result.V[i],$"HMMA low mismatch width={N} trial={trial} lane={i}");
                Check(d1[i]==HmmaState.Args[17].V[i],$"HMMA high mismatch width={N} trial={trial} lane={i}");
                checkedPairs+=2;
            }
        }
        Console.WriteLine($"PASS {checkedPairs} packed-half outputs bit-exact; actual emitters plus packed nonfinite transports, 32/64/128 lanes");
    }
}
'''
class HmmaShuffle(unittest.TestCase):
    def test_actual_emitters(self):
        dotnet=os.environ.get('DOTNET') or shutil.which('dotnet')
        if not dotnet: self.skipTest('requires .NET 10; set DOTNET')
        parent=ROOT/'target/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='hmma-shuffle-',dir=parent) as tmp:
            out=Path(tmp)
            (out/'test.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><EnableNETAnalyzers>false</EnableNETAnalyzers></PropertyGroup></Project>')
            (out/'Program.cs').write_text(mock+old+new+checks)
            run=subprocess.run([dotnet,'run','--project',str(out/'test.csproj'),'-c','Release','--verbosity','quiet'],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS',run.stdout)
            print(run.stdout.strip())

if __name__=='__main__': unittest.main()
