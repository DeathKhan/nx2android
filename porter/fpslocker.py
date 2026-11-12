# 60 fps from the community's FPSLocker configs (`porter port --fps60`): masagrator's FPSLocker-Warehouse keeps, per game build, the memory writes that lift a game's
# own frame limit on a Switch (FORMAT.md there). The port fetches the one config of the build it packs, and converts it for the host:
#   ALL_FPS write / evaluate_write / compare / evaluate_compare at MAIN addresses → fpslock.tsv in the data pack, which hle_fpslock.c reads at launch (DM_FPS_LOCKS) and
#     writes into game memory on every present, as it does its own rows. The expressions depend only on the frame rate target, so they are evaluated here for 30 and 60
#   MASTER_WRITE bytes → CUTS of a patches file for the rewriter (patch_nso.py DM_PATCHES_EXTRA), with the original bytes taken from the game's main
#   block timing: dropped (it switches off FPSLocker's own frame delay, which the host does not have)
# A config that uses anything else (asm_a64, DECLARATIONS, VARIABLE / HEAP / ALIAS addresses, refresh_rate) is refused whole: its entries work together, and half of them
# can leave the game worse off than at 30 fps. Of the 1,462 configs on 2026-10-28, 1,160 convert.
# The warehouse has no license: its configs are fetched for the player's own build at port time and never kept in the repository.
import ast, math, re, struct, urllib.error, urllib.request
from pathlib import Path

RAW = 'https://raw.githubusercontent.com/masagrator/FPSLocker-Warehouse/v4/SaltySD/plugins/FPSLocker/patches/{tid}/{bid}.yaml'
TARGETS = (30, 60)
REFRESH = 60   # the display rate FPS_LOCK_TARGET compares with (the host presents at 60 Hz)
TYPES = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q', 'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q', 'float': 'f', 'double': 'd'}
OPS = ('<', '<=', '>', '>=', '==', '!=')


class Refused(Exception): pass


def url(title_id, build_id): return RAW.format(tid=title_id.upper(), bid=build_id[:16].upper())


def fetch(title_id, build_id, timeout=30):
    """The config of this build as text, None when the warehouse has none (HTTP 404). Other errors (no network) raise OSError"""
    try:
        with urllib.request.urlopen(urllib.request.Request(url(title_id, build_id), headers={'User-Agent': 'nup2android'}), timeout=timeout) as r: return r.read().decode('utf-8-sig')
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        raise


def nso_image(path):
    """main's memory image (text, ro, data at their offsets; no bss) and its build ID (32 hex digits), as patch_nso.py reads it"""
    import lz4.block
    d = Path(path).read_bytes()
    if d[:4] != b'NSO0': raise Refused(f'{path} is not an NSO')
    flags = struct.unpack_from('<I', d, 0xc)[0]; segs = [struct.unpack_from('<III', d, 0x10 + i * 0x10) for i in range(3)]; csz = struct.unpack_from('<III', d, 0x60)
    img = bytearray(segs[2][1] + segs[2][2])
    for i, (fo, mo, sz) in enumerate(segs):
        raw = d[fo:fo + csz[i]]
        img[mo:mo + sz] = lz4.block.decompress(raw, uncompressed_size=sz) if flags >> i & 1 else raw
    return bytes(img), d[0x40:0x50].hex()


# ---- expressions (TinyExpr's syntax: ^ is a power) over FORMAT.md's globals ----
def globals_for(t):
    return {'FPS_TARGET': t, 'FRAMETIME_TARGET': 1000 / t, 'VSYNC_TARGET': max(1, int(REFRESH / t)), 'FPS_LOCK_TARGET': t + 2 if t == REFRESH else t}
FUNCS = {'TruncDec': lambda v, d: math.trunc(v * 10 ** int(d)) / 10 ** int(d), 'floor': math.floor, 'ceil': math.ceil, 'abs': abs, 'sqrt': math.sqrt, 'pow': math.pow,
         'fmod': math.fmod, 'log': math.log, 'log10': math.log10, 'exp': math.exp}


def evaluate(expr, t):
    if isinstance(expr, bool): raise Refused(f'not a number: {expr!r}')
    if isinstance(expr, (int, float)): return expr
    g = globals_for(t)
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return n.value
        if isinstance(n, ast.Name) and n.id in g: return g[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)): v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            for k, f in ((ast.Add, lambda: a + b), (ast.Sub, lambda: a - b), (ast.Mult, lambda: a * b), (ast.Div, lambda: a / b), (ast.Pow, lambda: a ** b), (ast.Mod, lambda: math.fmod(a, b))):
                if isinstance(n.op, k): return f()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords: return FUNCS[n.func.id](*[ev(x) for x in n.args])
        raise Refused(f'expression {expr!r}: {ast.unparse(n)} is not supported')
    try: tree = ast.parse(str(expr).replace('^', '**'), mode='eval')
    except SyntaxError: raise Refused(f'expression {expr!r} does not parse') from None
    return ev(tree.body)


def values(v, kind, t, evaluated):   # the value list of an entry at target t, as numbers of its type
    out = []
    for x in (v if isinstance(v, list) else [v]):
        n = evaluate(x, t) if evaluated else x
        if isinstance(n, bool) or not isinstance(n, (int, float)): raise Refused(f'value {x!r} is not a number')
        n = float(n) if kind in ('float', 'double') else int(n)   # FPSLocker converts the double to the type as C does: integers truncated
        try: struct.pack('<' + TYPES[kind], n)
        except struct.error: raise Refused(f'value {x!r} = {n} does not fit {kind}') from None
        out.append(n)
    return out


def fmt(vals, kind): return ','.join(repr(v) if kind in ('float', 'double') else str(v) for v in vals)


def chain(a, what):   # [MAIN, off, …]: every offset but the last names a pointer (FORMAT.md)
    if not isinstance(a, list) or len(a) < 2: raise Refused(f'{what} {a!r}: not [MAIN, offset, …]')
    if a[0] != 'MAIN': raise Refused(f'{what} in region {a[0]}: only MAIN addresses are supported')
    if not all(isinstance(o, int) and not isinstance(o, bool) for o in a[1:]) or a[1] < 0: raise Refused(f'{what} {a!r}: offsets must be numbers, the first not negative')
    return ','.join(f'{o:#x}' if o >= 0 else f'-{-o:#x}' for o in a[1:])   # after a pointer an offset may be negative ([MAIN, …, -4]: 15 configs on 2026-10-28)


def parse(text):
    import yaml
    text = re.sub(r'(?m)([\[,]\s*):([A-Za-z_]\w*)', r'\1":\2"', text)   # asm labels (:goto1) are plain scalars to FPSLocker's parser, a syntax error to PyYAML
    try: y = yaml.safe_load(text)
    except yaml.YAMLError as e: raise Refused(f'the config does not parse: {str(e).splitlines()[0]}') from None
    if not isinstance(y, dict): raise Refused('the config is not a YAML mapping')
    return y


def convert(text, image, build_id, source=''):
    """The config → (fpslock.tsv text, CUTS rows [(offset, original hex, new hex, reason)]). Raises Refused naming what the host cannot do"""
    y = parse(text)
    unknown = set(y) - {'ALL_FPS', 'MASTER_WRITE', 'DECLARATIONS', 'unsafeCheck', 'Addons'}
    if unknown: raise Refused(f'unknown sections {", ".join(sorted(map(str, unknown)))}')
    if y.get('DECLARATIONS'): raise Refused('it declares variables or code (DECLARATIONS), which the host has no buffer for')
    if y.get('Addons'): raise Refused('it needs extra files (Addons)')
    rows, cuts = [], []
    for i, e in enumerate(y.get('ALL_FPS') or []):
        what = f'ALL_FPS entry {i + 1}'
        if not isinstance(e, dict): raise Refused(f'{what} is not a mapping')
        t = e.get('type')
        if t == 'block':
            if e.get('what') != 'timing': raise Refused(f'{what}: block {e.get("what")!r}')
            continue
        if t not in ('write', 'evaluate_write', 'compare', 'evaluate_compare'): raise Refused(f'{what}: type {t!r}')
        kind = e.get('value_type')
        if kind == 'refresh_rate': raise Refused(f'{what}: it sets the display rate (refresh_rate), which the host keeps at 60 Hz')
        if kind not in TYPES: raise Refused(f'{what}: value type {kind!r}')
        ev = t.startswith('evaluate')
        v = {tg: fmt(values(e.get('value'), kind, tg, ev), kind) for tg in TARGETS}
        cmp = '-'
        if t.endswith('compare'):
            op, ck = e.get('compare_type'), e.get('compare_value_type')
            if op not in OPS or ck not in TYPES: raise Refused(f'{what}: compare {op!r} {ck!r}')
            cv = values(e.get('compare_value'), ck, 60, False)
            if len(cv) != 1: raise Refused(f'{what}: compare value {e.get("compare_value")!r}')
            cmp = f'{ck} {chain(e.get("compare_address"), what)} {op} {fmt(cv, ck)}'
        rows.append(f'{what}\t{kind}\t{chain(e.get("address"), what)}\t{v[30]}\t{v[60]}\t{cmp}')
    for i, e in enumerate(y.get('MASTER_WRITE') or []):
        what = f'MASTER_WRITE entry {i + 1}'
        if not isinstance(e, dict) or e.get('type') != 'bytes': raise Refused(f'{what}: type {e.get("type") if isinstance(e, dict) else e!r} (only bytes)')
        off, kind = e.get('main_offset'), e.get('value_type')
        if not isinstance(off, int) or kind not in TYPES: raise Refused(f'{what}: main_offset {off!r}, value type {kind!r}')
        new = b''.join(struct.pack('<' + TYPES[kind], n) for n in values(e.get('value'), kind, 60, False))
        if off + len(new) > len(image): raise Refused(f'{what}: main+{off:#x} is past main\'s image')
        old = image[off:off + len(new)]
        if old != new: cuts.append((off, old.hex(), new.hex(), f'FPSLocker-Warehouse {what}'))
    if not rows and not cuts: raise Refused('it writes nothing the host would apply')
    head = [f'# Written by nup2android\'s porter from the FPSLocker-Warehouse config {source} (masagrator and contributors); read by hle_fpslock.c (DM_FPS_LOCKS)',
            '# entry <TAB> value type <TAB> main offsets (all but the last name a pointer) <TAB> values at 30 fps <TAB> at 60 fps <TAB> compare (type, offsets, operator, value; - = always)',
            f'build\t{build_id}']
    return '\n'.join(head + rows) + '\n', cuts


def patches_py(build_id, cuts, source=''):   # the CUTS as a patches file for patch_nso.py (DM_PATCHES_EXTRA)
    body = ''.join(f'    ({o:#x}, {a!r}, {b!r}, {w!r}),\n' for o, a, b, w in cuts)
    return f'# Written by nup2android\'s porter from the FPSLocker-Warehouse config {source}: its MASTER_WRITE bytes (porter/fpslocker.py)\nCUTS = {{{build_id!r}: [\n{body}]}}\n'
