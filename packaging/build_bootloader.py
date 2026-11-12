# Compile PyInstaller's bootloader (the launcher nup2android.exe is made from) on this PC and put it in place of the prebuilt one in the installed
# PyInstaller. A few antivirus engines match the prebuilt launcher, which every PyInstaller app shares, and flag the bundle; a locally compiled one is a
# different binary (packaging/INSTALL.md). build_bundle.py refuses to build with the prebuilt one.
#   python packaging/build_bootloader.py      (needs the Microsoft C++ Build Tools: winget install Microsoft.VisualStudio.2022.BuildTools, workload VCTools)
# Run it again after installing or upgrading PyInstaller: pip puts the prebuilt bootloader back.
import hashlib, json, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

import PyInstaller

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'build/bootloader'
INSTALLED = Path(PyInstaller.__file__).parent / 'bootloader/Windows-64bit-intel'
MARKER = INSTALLED / 'nup2android-local-build.txt'   # sha256 of each launcher this script installed
LAUNCHERS = ('run.exe', 'runw.exe', 'run_d.exe', 'runw_d.exe')


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def is_local():
    """True when the installed launchers are the ones this script compiled."""
    if not MARKER.exists(): return False
    want = dict(line.split() for line in MARKER.read_text(encoding='utf-8').splitlines() if line.strip())
    return all((INSTALLED / n).exists() and want.get(n) == sha256(INSTALLED / n) for n in LAUNCHERS)


def main():
    ver = PyInstaller.__version__
    shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True)
    # the source of the installed version, from PyPI, checked against the sha256 PyPI gives
    rel = json.load(urllib.request.urlopen(f'https://pypi.org/pypi/pyinstaller/{ver}/json'))
    f, = [u for u in rel['urls'] if u['packagetype'] == 'sdist']
    sdist = WORK / f['filename']; sdist.write_bytes(urllib.request.urlopen(f['url']).read())
    if sha256(sdist) != f['digests']['sha256']: sys.exit(f'{sdist.name}: sha256 differs from PyPI')
    with tarfile.open(sdist) as t: t.extractall(WORK, filter='data')
    src = WORK / sdist.name.removesuffix('.tar.gz')
    # waf finds the Build Tools itself and installs the launchers into the source tree's PyInstaller/bootloader/
    subprocess.run([sys.executable, './waf', 'distclean', 'all', '--target-arch=64bit'], cwd=src / 'bootloader', check=True)
    built = src / 'PyInstaller/bootloader/Windows-64bit-intel'
    for n in LAUNCHERS: shutil.copy2(built / n, INSTALLED / n)
    MARKER.write_text(''.join(f'{n} {sha256(INSTALLED / n)}\n' for n in LAUNCHERS), encoding='utf-8')
    print(f'PyInstaller {ver}: compiled bootloader installed in {INSTALLED}')


if __name__ == '__main__':
    main()
