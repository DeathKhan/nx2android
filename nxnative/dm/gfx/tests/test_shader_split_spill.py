"""shader_split.py's pool results too large for one pipe message come back through a file (droidplay-core#307: Pokémon Pokopia's split failed on Windows with WinError 87,
its one big .fdata's result about 4.4 GB). The Mac sends such a message, so a worker here refuses any message over 1 MiB as Windows refuses a multi-GB one."""
import concurrent.futures as cf
import contextlib
import io
import multiprocessing as mp
import multiprocessing.connection
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import shader_split as split

LIMIT = 1 << 20


def fake_scan(job):   # scan_file's result shape: an even job is a file of 3 shaders with 1 MiB of code each, an odd one of 3 tiny ones
    n = LIMIT if job % 2 == 0 else 16
    uniq = {f'k{job}.{i}': (f'/f{job}', 'ps', bytes([job, i]) * 8, bytes([i]) * n) for i in range(3)}
    return f'/f{job}', uniq, {(f'k{job}.0', f'k{job}.1')}, None


def windows_worker(spill_dir):   # pool initializer: this worker can't send a message over LIMIT (Windows' WriteFile refusal is ERROR_INVALID_PARAMETER, errno 22)
    send = multiprocessing.connection.Connection._send_bytes
    def refuse_large(self, buf):
        if len(buf) > LIMIT: raise OSError(22, 'The parameter is incorrect')
        return send(self, buf)
    multiprocessing.connection.Connection._send_bytes = refuse_large
    split.scan_file, split.SPILL = fake_scan, LIMIT // 2
    split._pool_init(None, {}, {}, spill_dir)


def pool(spill_dir):
    return cf.ProcessPoolExecutor(2, mp_context=mp.get_context('spawn'), initializer=windows_worker, initargs=(spill_dir,))


class SpillTests(unittest.TestCase):
    def test_a_result_too_large_to_send_comes_back_through_a_file(self):
        with tempfile.TemporaryDirectory() as spill:
            with pool(spill) as ex, contextlib.redirect_stdout(io.StringIO()):
                results = split.pooled(ex, list(range(6)), 6, chunk=2)
            self.assertEqual(os.listdir(spill), [])   # each file is removed once read
        self.assertEqual(results, [fake_scan(j) for j in range(6)])   # in job order, as if sent

    def test_without_a_spill_folder_the_large_result_fails_as_in_the_report(self):
        with pool(None) as ex, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError) as e: split.pooled(ex, [0], 1)
        self.assertEqual(e.exception.errno, 22)

    def test_only_a_batch_over_the_limit_is_spilled(self):
        with tempfile.TemporaryDirectory() as spill, mock.patch.multiple(split, scan_file=fake_scan, SPILL=LIMIT // 2, SPILL_DIR=spill):
            self.assertEqual(split.scan_batch([1, 3]), [fake_scan(1), fake_scan(3)])   # 6 small shaders: sent as before
            path = split.scan_batch([1, 0])
            self.assertIsInstance(path, str)
            self.assertEqual(split.unspill(path), [fake_scan(1), fake_scan(0)])
            self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
