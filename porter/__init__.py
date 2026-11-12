# porter: the port pipeline. A player's game as one NUP container (already decrypted files) + a game profile → an APK and the game's data pack (docs/ARCHITECTURE.md).
#   python -m porter doctor | profiles | inspect <folder> | port <game.nup> | install <data pack> | devices | config
__version__ = '1.1.1'
