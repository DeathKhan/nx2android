# Where things are: the repository (and the nxnative subtree in it), the player's own folder (settings, the APK signing key).
import os, sys
from pathlib import Path

REPO = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent   # the bundle: nxnative/ beside nup2android.exe
NXNATIVE = REPO / 'nxnative'
DM = NXNATIVE / 'dm'
LOCAL_TOOLS = REPO / 'tools'   # a developer's toolchains next to the repository (not committed); the player installs their own


OLD_NAME = 'nx' + '2android'   # the folder before the rename: kept in use where it exists (its toolchains, settings and APK signing key)


def user_dir():
    """The player's nup2android folder: %APPDATA%\\nup2android on Windows, ~/Library/Application Support/nup2android on a Mac, ~/.config/nup2android elsewhere.
    A folder from before the rename is used as it is. NX2A_HOME overrides it (tests)"""
    if os.environ.get('NX2A_HOME'): d = Path(os.environ['NX2A_HOME'])
    else:
        if os.name == 'nt': base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData/Roaming')
        elif sys.platform == 'darwin': base = Path.home() / 'Library/Application Support'
        else: base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
        d = base / 'nup2android'
        if not d.exists() and (base / OLD_NAME).is_dir(): d = base / OLD_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def player_tools():   # the toolchains `porter tools --download` fetched for the player: android-ndk-r27c/, dotnet/, platform-tools/
    return user_dir() / 'toolchains'


def signing_key():   # one key per player: every app they port keeps its signer, so a new port installs over the last one
    return user_dir() / 'signing-key.json'
