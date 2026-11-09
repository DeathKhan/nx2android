#!/usr/bin/env python3
import ipaddress,re,subprocess,sys
from pathlib import Path
host=str(ipaddress.ip_address(sys.argv[1]))
source=Path('/home/james/Projects/nx2android/shim/adb').read_text()
source,n=re.subn(r'(?m)^HOST=[^;]+;', 'HOST='+host+';',source,count=1)
if n != 1: raise SystemExit('Could not locate transport host')
if ':' in host: source=source.replace('$USER@$HOST:', '$USER@[$HOST]:')
raise SystemExit(subprocess.run(['bash','-s','--',*sys.argv[2:]],input=source.encode()).returncode)
