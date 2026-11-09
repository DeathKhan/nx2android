Task-owned optional transport wrapper; no device action has been performed by preparation/tests.
Use: python3 target/adb-transport/adb_keepalive.py DEVICE_IP adb-command [arguments...]
Fallback: python3 target/adb-host.py DEVICE_IP adb-command [arguments...]

Only the wrapped helper's child PATH changes. Bare ssh/scp in the canonical adb source
resolve through these wrappers. The unchanged adb-host.py reads the canonical source
and credentials into memory; this directory contains no credentials or source copy.
ssh receives ServerAliveInterval=10, ServerAliveCountMax=3, ControlMaster=auto,
ControlPersist=300. scp uses -S with this exact ssh wrapper. Existing command args
remain unchanged. Multiplexing is per supplied IP; IPv4/IPv6 addresses use distinct
12-hex hash sockets under target/ct, mode0700. That short path also leaves room for
OpenSSH's temporary socket suffix. These options do not repair an unreachable server.

Cleanup after the test: stop using the wrapper and use the original helper.
Owned masters expire after300 seconds idle. To close them earlier, root may issue
/usr/bin/ssh -S SOCKET -O exit ignored
for sockets in this task's target/ct only; this closes the task connection, not sshd.
After master shutdown, remove target/adb-transport and target/ct only.
No ~/.ssh/config, canonical shim, credentials, services or production files changed.
