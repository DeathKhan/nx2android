#!/usr/bin/env python3
# darwin2elf.py <in.S> <out.S>: an AArch64 assembly file written for Mach-O (the host's entry.S, nvn_entry.S, the generated nvn_thunks.S) as ELF assembly
# for the Android build. C symbols lose Mach-O's leading underscore, adrp / add / ldr operands sym@PAGE / sym@PAGEOFF become sym / :lo12:sym,
# .private_extern becomes .hidden, and Mach-O's section shorthands become ELF sections. Local labels (1:, 1f) and numbers are untouched.
# A symbol is "C" when it starts with one underscore followed by a letter.
import re, sys

src, dst = sys.argv[1], sys.argv[2]
SYM = r'_([A-Za-z][A-Za-z0-9_$.]*)'
SECTIONS = [(r'\.cstring\b', '.section .rodata.str1.1,"aMS",@progbits,1'), (r'\.const_data\b', '.section .data.rel.ro'), (r'\.const\b', '.section .rodata')]
out = []
for line in open(src, encoding='utf-8'):
    code, sep, comment = line.partition('//')
    code = re.sub(SYM + r'@PAGEOFF', r':lo12:\1', code)
    code = re.sub(SYM + r'@PAGE', r'\1', code)
    code = code.replace('.private_extern', '.hidden')
    m = re.match(r'^(\s*)\.zerofill\s+__DATA\s*,\s*__bss\s*,\s*' + SYM + r'\s*,\s*(\d+)\s*,\s*(\d+)\s*$', code.rstrip('\n'))
    if m:   # .zerofill __DATA,__bss,_sym,size,align (align as a power of two) -> an object in .bss; the section in force before it is restored
        ind, sym, size, align = m.groups()
        code = f'{ind}.section .bss\n{ind}.hidden {sym}\n{ind}.p2align {align}\n{ind}{sym}:\n{ind}.skip {size}\n{ind}.previous\n'   # hidden: adrp reaches it directly in the shared library (as the Mach-O build does)
    for pat, rep in SECTIONS: code = re.sub(r'^(\s*)' + pat, lambda m: m.group(1) + rep, code)
    code = re.sub(r'(\.globl\s+|\.hidden\s+|\b(?:bl|b|b\.\w+|cbz\s+\w+,|cbnz\s+\w+,|tbz\s+\w+,\s*#\d+,|tbnz\s+\w+,\s*#\d+,|adr\s+\w+,)\s*)' + SYM + r'\b', r'\1\2', code)
    code = re.sub(r'^(\s*)' + SYM + r':', r'\1\2:', code)   # label definitions
    code = re.sub(r'(\.(?:quad|word|xword|8byte)\s+)' + SYM + r'\b', r'\1\2', code)   # data references
    out.append(code + sep + comment)
text = ''.join(out)
if '@PAGE' in text: sys.exit(f'{src}: unconverted @PAGE operand')
open(dst, 'w', encoding='utf-8', newline='\n').write(text)
