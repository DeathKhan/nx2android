#!/usr/bin/env python3
"""Rebuild an Android shader bundle through the porter's source compiler.

All developer outputs stay under this checkout's dist/target. The Mac pinned
translator/Metal lock is untouched. Example:
  python devtools/build_shader_bundle.py PKG dist/shaders.dmsv --game ID \
    --profile /path/to/profile/ID --dotnet /path/to/dotnet --validate \
    --baseline /path/to/installed/shaders.dmsv --hmma-shuffle 1
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from porter.pipeline import build_translator

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('package',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--game',required=True);p.add_argument('--profile',type=Path)
    p.add_argument('--dotnet',default=os.environ.get('DOTNET') or shutil.which('dotnet'))
    p.add_argument('--jobs',type=int,default=8);p.add_argument('--keys',default='')
    p.add_argument('--capture',type=Path)
    p.add_argument('--hmma-shuffle',choices=('0','1'))
    p.add_argument('--validate',action='store_true')
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if not a.dotnet:p.error('provide --dotnet or DOTNET (.NET10)')
    if a.jobs<1:p.error('--jobs must be positive')
    if a.validate and not shutil.which('spirv-val'):p.error('--validate needs spirv-val')
    out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    target=ROOT/'target';target.mkdir(exist_ok=True)
    tools=ROOT/'dist/android-shader-tools';tools.mkdir(parents=True,exist_ok=True)
    logs=tools/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,DM_GAME=a.game,DOTNET_ROOT=str(Path(a.dotnet).resolve().parent))
    if a.profile:
        profile=a.profile.resolve()
        if profile.name!=a.game or not (profile/'game.tsv').is_file():p.error('--profile must name the game-ID directory containing game.tsv')
        env.update(DM_GAMES_DIR=str(profile.parent),DM_GAMES_FIRST='1')
    if a.hmma_shuffle is not None:env['DMSH_HMMA_SHUFFLE']=a.hmma_shuffle
    compiler,compiler_key=build_translator(a.dotnet,tools/'dmsh_xlate',logs/'translator.log',env)
    temp=out.with_name(out.name+'.partial')
    script=ROOT/'nxnative/dm/android/tools/shader_vk.py'
    sys.path.insert(0,str(script.parent));import dmsv
    with tempfile.TemporaryDirectory(prefix='shader-bundle-',dir=target) as scratch:
        env['WORK']=scratch
        cmd=[sys.executable,str(script),str(a.package.resolve()),str(temp),'--jobs',str(a.jobs),'--dotnet',a.dotnet,'--xlate',str(compiler)]
        if a.keys:cmd+=['--keys',a.keys]
        if a.capture:cmd+=['--capture',str(a.capture.resolve())]
        log=out.with_name(out.name+'.build.log')
        with open(log,'w') as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        with dmsv.Bundle(temp) as bundle:
            if not bundle.entries:raise RuntimeError('translation produced no shaders; see '+str(log))
            report={'game':a.game,'entries':len(bundle.entries),'bundle_sha256':sha(temp),'bundle_bytes':temp.stat().st_size,
                    'compiler_key':compiler_key,'compiler':json.loads((compiler.parent/'build-manifest.json').read_text()),
                    'shader_builder_sha256':sha(script),'profile':str(a.profile.resolve()) if a.profile else None,
                    'hmma_shuffle':env.get('DMSH_HMMA_SHUFFLE','builder default')}
            if a.baseline:
                with dmsv.Bundle(a.baseline) as baseline:
                    old={e[0]:e for e in baseline.entries};new={e[0]:e for e in bundle.entries}
                    if old.keys()!=new.keys():raise RuntimeError(f'key set changed: missing {len(old.keys()-new.keys())}, extra {len(new.keys()-old.keys())}')
                    for key,e in new.items():
                        if old[key][1]!=e[1] or baseline.refl(old[key])!=bundle.refl(e):raise RuntimeError(f'reflection changed: {key:016x}')
                    report.update(baseline_sha256=sha(a.baseline),baseline_keys_and_reflection_identical=True)
            if a.validate:
                def validate(entry):
                    key=entry[0];path=Path(scratch)/f'{key:016x}.spv';path.write_bytes(bundle.spv(entry))
                    run=subprocess.run(['spirv-val','--target-env','vulkan1.2',str(path)],capture_output=True,text=True);path.unlink()
                    return {'key':f'{key:016x}','error':run.stderr} if run.returncode else None
                with cf.ThreadPoolExecutor(a.jobs) as ex:errors=[r for r in ex.map(validate,bundle.entries) if r]
                report.update(spirv_validation={'target':'vulkan1.2','passed':len(bundle.entries)-len(errors),'errors':errors})
                if errors:
                    out.with_name(out.name+'.validation-failed.json').write_text(json.dumps(report,indent=2)+'\n')
                    raise RuntimeError(f'{len(errors)} SPIR-V validation failures; final output preserved')
        os.replace(temp,out)
    out.with_name(out.name+'.evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'bundle':str(out),'entries':report['entries'],'sha256':report['bundle_sha256'],
                      'compiler_key':compiler_key,'validation':'all passed' if a.validate else 'not requested'},indent=2))

if __name__=='__main__':main()

