import concurrent.futures
from pathlib import Path
import struct
import tempfile
import unittest

import brepair


def model_bytes():
    tokens = [(bytes([n]), False) for n in range(256)]
    tokens += [(b'', True), (b'hello', False), (b' world', True), ('你好'.encode(), False)]
    return b'BRTOKN01'+struct.pack('<I', len(tokens))+b''.join(
        struct.pack('<IB', len(raw), end)+raw for raw,end in tokens)


class PythonApiTests(unittest.TestCase):
    def test_native_roundtrip_binary_unicode_empty_and_persistence(self):
        with brepair.Tokenizer.from_bytes(model_bytes()) as tokenizer:
            self.assertEqual(tokenizer.vocabulary_size, 260)
            self.assertEqual(tokenizer.encode(b'hello world'), [257,258])
            for raw in [b'', b'\0\xffhello\r\n', '你好世界'.encode()]:
                self.assertEqual(tokenizer.decode(tokenizer.encode(raw)), raw)
            self.assertEqual(tokenizer.encode('你好'), tokenizer.encode('你好'.encode()))
            self.assertEqual(tokenizer.to_bytes(), model_bytes())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'词汇.brt'
                tokenizer.save(path)
                with brepair.Tokenizer(path) as restored:
                    self.assertEqual(restored.encode('你好'), tokenizer.encode('你好'))
                with self.assertRaises(FileExistsError):
                    tokenizer.save(path)

    def test_errors_and_limits_are_python_exceptions(self):
        with self.assertRaises(ValueError):
            brepair.Tokenizer.from_bytes(b'bad model')
        with brepair.Tokenizer.from_bytes(model_bytes()) as tokenizer:
            for ids in [[], [97], [256,97], [0xffffffff], [-1], [1 << 32]]:
                with self.assertRaises(ValueError):
                    tokenizer.decode(ids)
            with self.assertRaises(TypeError):
                tokenizer.decode([1.5])
            with self.assertRaises(ValueError):
                tokenizer.decode(tokenizer.encode('hello'), maximum_bytes=4)
            tokenizer.close()
            tokenizer.close()
            with self.assertRaises(RuntimeError):
                tokenizer.encode(b'closed')

    def test_forks_share_model_keep_lifetime_and_own_workspaces(self):
        source = brepair.Tokenizer.from_bytes(model_bytes())
        workers = [source.fork() for _ in range(4)]
        source.close()
        def run(worker):
            with worker:
                for i in range(100):
                    raw = f'hello world {i}'.encode()
                    self.assertEqual(worker.decode(worker.encode(raw)), raw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run, workers))

    def test_file_discovery_selection_and_huffman_workflows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'train.txt'
            source.write_bytes((b'hello world\nhello there\n'*256)+b'\0\xff\n')
            result = brepair.train_file(source, root/'candidates.brp', minimum_count=2,
                maximum_rules=64, threads=2, memory='128MB')
            self.assertIn('train', result.command)
            brepair.select_vocabulary(root/'candidates.brp', source, root/'vocabulary.brt',
                vocabulary_size=260, minimum_uses=1, threads=2, memory='128MB')
            with brepair.Tokenizer(root/'vocabulary.brt') as tokenizer:
                for field in source.read_bytes().splitlines(keepends=True):
                    self.assertEqual(tokenizer.decode(tokenizer.encode(field)), field)
            brepair.compress_file(source, root/'archive.brh', threads=2, memory='128MB')
            brepair.decompress_file(root/'archive.brh', root/'restored.txt', memory='128MB')
            self.assertEqual((root/'restored.txt').read_bytes(), source.read_bytes())
            brepair.decompress_file(root/'candidates.brp', root/'training-restored.txt', memory='128MB')
            self.assertEqual((root/'training-restored.txt').read_bytes(), source.read_bytes())

    def test_independent_columns_projection_and_exact_row_restoration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            raw=b'0001|"quoted"|\\N\r\n|0|\xff\n7|8|final'
            source=root/'source.psv';source.write_bytes(raw)
            result=brepair.compress_columns(source,root/'columns',columns=3,threads=2,memory='128MB')
            self.assertEqual(result['rows'],3)
            self.assertEqual(result['complete_bytes'],sum(p.stat().st_size for p in (root/'columns').iterdir()))
            brepair.decompress_column(root/'columns',1,root/'projected.txt',memory='128MB')
            self.assertEqual((root/'projected.txt').read_bytes(),b'"quoted"\n0\n8\n')
            brepair.decompress_columns(root/'columns',root/'restored.psv',memory='128MB')
            self.assertEqual((root/'restored.psv').read_bytes(),raw)
            with self.assertRaises(FileExistsError):
                brepair.decompress_columns(root/'columns',root/'restored.psv')


if __name__ == '__main__':
    unittest.main()
