#!/usr/bin/env python3
"""Wrap the actual translated shared/shuffle HMMA functions in one GPU A/B kernel.

Usage: hmma_compare_spv.py corrected.spv optimized.spv output.spv
No game instructions run: the original main is replaced by an SSBO input/output
wrapper. Both real helper functions receive identical packed registers.
"""
import re
import subprocess
import sys
from pathlib import Path

def dis(path):
    return subprocess.check_output(['spirv-dis',str(path)],text=True)
def func(text,name):
    m=re.search(r'^\s*'+re.escape(name)+r' = OpFunction\b',text,re.M)
    end=text.index('OpFunctionEnd',m.end())+len('OpFunctionEnd')
    return m.start(),end,text[m.start():end]
def defs(text):
    return dict(re.findall(r'^\s*(%\w+) = (.*)$',text,re.M))
def main(oldpath,newpath,outpath):
    old,new=dis(oldpath),dis(newpath)
    name='%dmsh_hmma16816_f16'
    _,_,ofn=func(old,name);_,_,nfn=func(new,name)
    assert ofn.count('OpControlBarrier')==2
    assert nfn.count('OpControlBarrier')==0 and nfn.count('OpGroupNonUniformShuffle')==32
    assert ofn.count(' Fma ')==nfn.count(' Fma ')==64
    oldglobals=old[:old.index(' = OpFunction ')].rsplit('\n',1)[0]
    newglobals=new[:new.index(' = OpFunction ')].rsplit('\n',1)[0]
    od,nd=defs(oldglobals),defs(newglobals)
    local=defs(nfn)
    imported=[]
    mapping={x:('%hmma_shuffle' if x==name else '%opt_'+x[1:]) for x in local}
    def resolve(x):
        if x in mapping:return mapping[x]
        rhs=nd[x]
        rhs=re.sub(r'%\w+',lambda m:resolve(m[0]),rhs)
        # Prefer the same named builtin/global before structural interning.
        match=x if od.get(x)==rhs else next((k for k,v in od.items() if v==rhs),None)
        if match is None:
            match='%hmma_import_'+str(len(imported))
            imported.append(f'{match} = {rhs}');od[match]=rhs
        mapping[x]=match
        return match
    nfn=re.sub(r'%\w+',lambda m:resolve(m[0]),nfn)
    params=re.findall(r'^\s*(%\w+) = OpFunctionParameter',ofn,re.M)
    assert len(params)==18
    mainid=re.search(r'OpEntryPoint GLCompute (%\w+)',old)[1]
    start,end,oldmain=func(old,mainid)
    mainhead=oldmain.splitlines()[0]
    globalnew=[]
    def g(name,rhs):
        found=next((k for k,v in od.items() if v==rhs),None)
        if found:return found
        key='%test_'+name;od[key]=rhs;globalnew.append(f'{key} = {rhs}');return key
    uint=next(k for k,v in od.items() if v=='OpTypeInt 32 0')
    sint=next(k for k,v in od.items() if v=='OpTypeInt 32 1')
    pin=g('pint',f'OpTypePointer Function {sint}')
    cin=lambda n:g('i'+str(n),f'OpConstant {sint} {n}')
    cu=lambda n:g('u'+str(n),f'OpConstant {uint} {n}')
    # 128 workgroups *128 invocations *14 words (8 input,4 output,2 subgroup metadata).
    arr=g('array',f'OpTypeArray {uint} {cu(128*128*14)}')
    st=g('block',f'OpTypeStruct {arr}')
    ptr=g('ptrblock',f'OpTypePointer StorageBuffer {st}')
    pw=g('pword',f'OpTypePointer StorageBuffer {uint}')
    pinput=g('pinput',f'OpTypePointer Input {uint}')
    buffer='%test_buffer';globalnew.append(f'{buffer} = OpVariable {ptr} StorageBuffer')
    sgsize='%test_subgroup_size';globalnew.append(f'{sgsize} = OpVariable {pinput} Input')
    decor=f'''OpDecorate {arr} ArrayStride 4
OpDecorate {st} Block
OpMemberDecorate {st} 0 Offset 0
OpDecorate {buffer} DescriptorSet 0
OpDecorate {buffer} Binding 0
OpDecorate {sgsize} BuiltIn SubgroupSize
'''
    def builtin(what):
        return re.search(r'OpDecorate (%\w+) BuiltIn '+what+r'\b',old)[1]
    b_local=builtin('LocalInvocationId');b_group=builtin('WorkgroupId');b_lane=builtin('SubgroupLocalInvocationId')
    type_local=od[od[b_local].split()[1]].split()[-1]
    lines=[mainhead,'%test_label = OpLabel']
    args=[]
    for i in range(18):
        a=f'%test_arg{i}';args.append(a);lines.append(f'{a} = OpVariable {pin} Function')
    counter=0
    def emit(op):
        nonlocal counter
        key='%test_val'+str(counter);counter+=1;lines.append(f'{key} = {op}');return key
    lc=emit(f'OpLoad {type_local} {b_local}');lx=emit(f'OpCompositeExtract {uint} {lc} 0')
    wg=emit(f'OpLoad {type_local} {b_group}');wx=emit(f'OpCompositeExtract {uint} {wg} 0')
    gi=emit(f'OpIMul {uint} {wx} {cu(128)}');gi=emit(f'OpIAdd {uint} {gi} {lx}')
    base=emit(f'OpIMul {uint} {gi} {cu(14)}')
    lane=emit(f'OpLoad {uint} {b_lane}')
    logical=emit(f'OpBitwiseAnd {uint} {lane} {cu(31)}');logical=emit(f'OpBitcast {sint} {logical}')
    block=emit(f'OpShiftRightLogical {uint} {lx} {cu(5)}');block=emit(f'OpShiftLeftLogical {uint} {block} {cu(8)}');block=emit(f'OpBitcast {sint} {block}')
    lines += [f'OpStore {args[0]} {logical}',f'OpStore {args[1]} {block}']
    def word(i):
        ix=emit(f'OpIAdd {uint} {base} {cu(i)}')
        return emit(f'OpAccessChain {pw} {buffer} {cu(0)} {ix}')
    for i in range(8):
        load=emit(f'OpLoad {uint} {word(i)}');load=emit(f'OpBitcast {sint} {load}')
        lines.append(f'OpStore {args[i+2]} {load}')
    for i in range(10,17):lines.append(f'OpStore {args[i]} {cin(0)}')
    for fn,offset in [(name,8),('%hmma_shuffle',10)]:
        first=emit(f'OpFunctionCall {sint} {fn} '+' '.join(args))
        second=emit(f'OpLoad {sint} {args[17]}')
        for j,value in enumerate([first,second]):
            u=emit(f'OpBitcast {uint} {value}');dst=word(offset+j);lines.append(f'OpStore {dst} {u}')
    actualsize=emit(f'OpLoad {uint} {sgsize}')
    for j,value in [(12,actualsize),(13,lane)]:dst=word(j);lines.append(f'OpStore {dst} {value}')
    lines += ['OpReturn','OpFunctionEnd']
    merged=old[:start]+'\n'.join(lines)+old[end:]+'\n'+nfn+'\n'
    merged=re.sub(r'(OpEntryPoint GLCompute [^\n]+)',lambda m:m[0]+' '+sgsize,merged,count=1)
    # Annotations precede types; declarations precede the first function.
    pos=re.search(r'^\s*%\w+ = OpType',merged,re.M).start()
    merged=merged[:pos]+decor+merged[pos:]
    pos=re.search(r'^\s*%\w+ = OpFunction ',merged,re.M).start()
    merged=merged[:pos]+'\n'.join(imported+globalnew)+'\n'+merged[pos:]
    outpath=Path(outpath);outpath.parent.mkdir(parents=True,exist_ok=True)
    asm=outpath.with_suffix('.spvasm');asm.write_text(merged)
    subprocess.run(['spirv-as','--target-env','spv1.3',str(asm),'-o',str(outpath)],check=True)
    subprocess.run(['spirv-val','--target-env','vulkan1.2',str(outpath)],check=True)
    # Keep actual helper instructions, dropping only unused original game code/resources.
    subprocess.run(['spirv-opt','--eliminate-dead-functions','--eliminate-dead-variables',str(outpath),'-o',str(outpath)],check=True)
    subprocess.run(['spirv-val','--target-env','vulkan1.2',str(outpath)],check=True)
    print(f'PASS actual shared/shuffle helpers wrapped: {outpath}, {outpath.stat().st_size} bytes')

if __name__=='__main__': main(*sys.argv[1:])

