# Downloading the toolchains a port needs, for a player who has none (`porter tools --download`, the Setup tab's Download button): the Android NDK r27c and
# platform-tools from Google's SDK repository, the newest .NET SDK 10 from Microsoft's release list. Each archive is checked against the checksum its list gives,
# then unpacked into the player's nup2android folder (paths.player_tools()), where toolchain.find_* look, and the settings point at it. The tool ships none of them:
# they come from their makers, under their licenses (LICENSES).
import hashlib, json, os, platform, shutil, stat, sys, tarfile, urllib.request, xml.etree.ElementTree as ET, zipfile
from dataclasses import dataclass
from pathlib import Path

from porter import config, toolchain
from porter.paths import player_tools
from porter.upstream import BuildError, PROG

GOOGLE = 'https://dl.google.com/android/repository/'
GOOGLE_INDEX = GOOGLE + 'repository2-3.xml'   # what sdkmanager reads: every package's archives, sizes and SHA-1 per host OS
DOTNET_INDEX = f'https://builds.dotnet.microsoft.com/dotnet/release-metadata/{toolchain.DOTNET_MAJOR}.0/releases.json'   # every release's files with SHA-512
LICENSES = {'Android NDK': 'https://developer.android.com/studio/terms', 'adb': 'https://developer.android.com/studio/terms',
            '.NET SDK': 'https://github.com/dotnet/core/blob/main/license-information.md'}
NDK_PACKAGE = f'ndk;{toolchain.NDK_REVISION}'
UNPACKED = {'Android NDK': 2.6 * 2**30, '.NET SDK': 0.9 * 2**30, 'adb': 0.05 * 2**30}   # disk space unpacked (r27c on Windows: 2.5 GB), for the free-space check


@dataclass
class Item:
    name: str      # the doctor's row: 'Android NDK', '.NET SDK', 'adb'
    version: str
    url: str
    size: int
    hash: str
    algo: str      # 'sha1' / 'sha512'
    folder: str    # under player_tools(): android-ndk-r27c, dotnet, platform-tools
    license: str

    def as_dict(self): return dict(self.__dict__)


def _get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'nup2android'}), timeout=timeout) as r: return r.read()


def _google_os(): return {'win32': 'windows', 'darwin': 'macosx'}.get(sys.platform, 'linux')


def _dotnet_rid():
    os_ = {'win32': 'win', 'darwin': 'osx'}.get(sys.platform, 'linux')
    return f'{os_}-{"arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"}'


def _google(index, path, name, folder):
    """The archive of SDK package `path` for this OS in Google's index"""
    for p in index.iter('remotePackage'):
        if p.get('path') != path: continue
        rev = p.find('revision'); version = '.'.join(rev.findtext(k) for k in ('major', 'minor', 'micro') if rev.findtext(k) is not None)
        for a in p.iter('archive'):
            if a.findtext('host-os') not in (None, _google_os()): continue
            c = a.find('complete')
            return Item(name, version, GOOGLE + c.findtext('url'), int(c.findtext('size')), c.findtext('checksum'), c.find('checksum').get('type', 'sha1'), folder, LICENSES[name])
    raise BuildError(f'{path} for {_google_os()} is not in Google\'s SDK repository ({GOOGLE_INDEX})')


def _dotnet():
    """The newest .NET SDK of the major the port builds with, for this OS and CPU"""
    rel = json.loads(_get(DOTNET_INDEX))
    rid, kind = _dotnet_rid(), ('.zip' if sys.platform == 'win32' else '.tar.gz')
    for r in rel['releases']:   # newest first
        for sdk in [r.get('sdk')] + r.get('sdks', []):
            for f in (sdk or {}).get('files', []):
                if f.get('rid') == rid and f['name'] == f'dotnet-sdk-{rid}{kind}':
                    return Item('.NET SDK', sdk['version'], f['url'], 0, f['hash'], 'sha512', 'dotnet', LICENSES['.NET SDK'])
    raise BuildError(f'no .NET SDK {toolchain.DOTNET_MAJOR} for {rid} in {DOTNET_INDEX}')


def missing(cfg=None):
    """The doctor's rows a download can fix: NDK r27c, .NET SDK 10, adb"""
    rows = {r.name: r for r in toolchain.checks(cfg)}
    return [n for n in ('Android NDK', '.NET SDK', 'adb') if not rows[n].ok]


def plan(names=None):
    """The Items to download: the missing tools (or `names`), with their sizes"""
    names = missing() if names is None else names
    out, index = [], None
    for n in names:
        if n in ('Android NDK', 'adb') and index is None: index = ET.fromstring(_get(GOOGLE_INDEX))
        if n == 'Android NDK': out.append(_google(index, NDK_PACKAGE, n, f'android-ndk-{toolchain.NDK_RELEASE}'))
        elif n == 'adb': out.append(_google(index, 'platform-tools', n, 'platform-tools'))
        elif n == '.NET SDK': out.append(_dotnet())
    for it in out:
        if not it.size:
            with urllib.request.urlopen(urllib.request.Request(it.url, method='HEAD', headers={'User-Agent': 'nup2android'}), timeout=60) as r: it.size = int(r.headers.get('Content-Length') or 0)
    return out


def _download(it, dest):   # to dest, counting bytes and hashing as it goes; a wrong checksum deletes the file
    h, part = hashlib.new(it.algo), dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(urllib.request.Request(it.url, headers={'User-Agent': 'nup2android'}), timeout=120) as r, open(part, 'wb') as f:
        total, done = int(r.headers.get('Content-Length') or it.size or 0), 0
        PROG.set(step='downloading', done=0, total=total or None, unit='B')
        while chunk := r.read(1 << 20):
            f.write(chunk); h.update(chunk); done += len(chunk); PROG.set(done=done)
    if h.hexdigest().lower() != it.hash.lower():
        part.unlink(missing_ok=True); raise BuildError(f'{it.url}: the download\'s {it.algo} is not the one its list gives (try again)')
    part.replace(dest)


def _unzip(arc, into):   # zipfile drops Unix modes and symlinks; the NDK and platform-tools need both outside Windows
    with zipfile.ZipFile(arc) as z:
        infos = z.infolist(); PROG.set(step='unpacking', done=0, total=len(infos), unit='files')
        for i, info in enumerate(infos):
            mode = info.external_attr >> 16
            if os.name != 'nt' and stat.S_ISLNK(mode):
                t = into / info.filename; t.parent.mkdir(parents=True, exist_ok=True); t.unlink(missing_ok=True); os.symlink(z.read(info).decode(), t)
            else:
                p = Path(z.extract(info, into))
                if os.name != 'nt' and mode & 0o777 and not info.is_dir(): p.chmod(mode & 0o777)
            if i % 200 == 0: PROG.set(done=i)


def _untar(arc, into):
    with tarfile.open(arc) as t:
        PROG.set(step='unpacking'); t.extractall(into, **({'filter': 'tar'} if hasattr(tarfile, 'tar_filter') else {}))   # the filter: Python 3.12+ (3.11.4)


def _install(it, arc):
    """Unpack arc into player_tools()/it.folder: an archive with one top folder (the NDK, platform-tools) gives that folder, one without (.NET) is the folder"""
    root = player_tools(); tmp = root / f'.{it.folder}.unpack'
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    (_unzip if arc.suffix == '.zip' else _untar)(arc, tmp)
    top = list(tmp.iterdir())
    src = top[0] if len(top) == 1 and top[0].is_dir() and it.folder != 'dotnet' else tmp
    dest = root / it.folder
    shutil.rmtree(dest, ignore_errors=True); src.replace(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def download(items):
    """Download, check and unpack every Item; the settings then point at them. One stage per tool"""
    root = player_tools(); root.mkdir(parents=True, exist_ok=True)
    need = sum(it.size + UNPACKED.get(it.name, 0) for it in items); free = shutil.disk_usage(root).free
    if free < need: raise BuildError(f'{need / 2**30:.1f} GB needed on {root.anchor} for the tools, {free / 2**30:.1f} GB free')
    PROG.plan(len(items))
    keys = {'Android NDK': ('ndk', ''), '.NET SDK': ('dotnet', f'dotnet{toolchain.EXE}'), 'adb': ('adb', f'adb{toolchain.EXE}')}
    for it in items:
        PROG.stage(f'{it.name} {it.version}', step='downloading')
        arc = root / Path(it.url).name
        try:
            _download(it, arc); dest = _install(it, arc)
        finally: arc.unlink(missing_ok=True)
        key, exe = keys[it.name]
        config.set_value(key, str(dest / exe) if exe else str(dest))
        PROG.end(str(dest))
    PROG.close()
    bad = [n for n in missing() if n in {it.name for it in items}]
    if bad: raise BuildError('downloaded, but the check still fails for: ' + ', '.join(bad) + ' (porter doctor says why)')
