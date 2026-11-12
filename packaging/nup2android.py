# The bundle's one executable (packaging/build_bundle.py). With no arguments it is the desktop app; otherwise it is the Python the tool runs its parts with,
# since a bundle has no python.exe:
#   nup2android -m porter …        the port process the GUI starts (gui/runner.py)
#   nup2android <script.py> …      the engine's tools the pipeline runs (patch_nso.py, shader_split.py, dm/android/build.py …), as `python <script.py>`
import multiprocessing, multiprocessing.spawn, os, runpy, sys


def main():
    multiprocessing.freeze_support()   # a tool's pool worker (shader_split.py) starts here and runs its work in there
    # A frozen Windows program does not send pool workers the main script's path (multiprocessing.spawn WINEXE), so a worker could not find the tool's
    # functions; with it, the worker loads the tool as __mp_main__ as an ordinary Python does
    multiprocessing.spawn.WINEXE = False
    a = sys.argv[1:]
    if a[:1] == ['-m'] and len(a) > 1:
        sys.argv = [a[1]] + a[2:]
        runpy.run_module(a[1], run_name='__main__', alter_sys=True)
    elif a and a[0].endswith('.py'):
        sys.argv = a
        sys.path.insert(0, os.path.dirname(os.path.abspath(a[0])))   # as python does: the script's folder first (the tools import their neighbours)
        runpy.run_path(a[0], run_name='__main__')
    else:
        if os.name == 'nt':   # opened by double-click, the console window is the app's alone: hide it (from a terminal it stays)
            import ctypes
            k = ctypes.windll.kernel32; ids = (ctypes.c_uint * 4)()
            if k.GetConsoleProcessList(ids, 4) == 1: ctypes.windll.user32.ShowWindow(k.GetConsoleWindow(), 0)
        from gui.app import main as gui
        sys.exit(gui(sys.argv))


if __name__ == '__main__':
    main()
