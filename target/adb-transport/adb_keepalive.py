#!/usr/bin/env python3
"""Task-only PATH wrappers; canonical adb credentials remain in the original helper's memory."""
import hashlib, ipaddress, os, subprocess, sys
from pathlib import Path
base = Path(__file__).resolve().parent
root = base.parent.parent
if len(sys.argv) < 3:
    raise SystemExit("usage: adb_keepalive.py DEVICE_IP adb-command [arguments...]")
host = str(ipaddress.ip_address(sys.argv[1]))
env = os.environ.copy()
env["CODEX_ADB_CONTROL_PATH"] = str(root / "target/ct" / hashlib.sha256(host.encode()).hexdigest()[:12])
env["PATH"] = str(base) + os.pathsep + env.get("PATH", os.defpath)
raise SystemExit(subprocess.run([sys.executable, str(root / "target/adb-host.py"), host, *sys.argv[2:]], env=env).returncode)
