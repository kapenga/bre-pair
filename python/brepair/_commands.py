from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import time


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    stdout: str
    stderr: str
    seconds: float


def _run(arguments, executable=None):
    bundled = Path(__file__).parent/'_native'/('brepair.exe' if os.name == 'nt' else 'brepair')
    selected = executable or os.environ.get('BREPAIR_EXECUTABLE') or (bundled if bundled.is_file() else shutil.which('brepair'))
    if selected is None:
        raise RuntimeError('BRe-Pair executable missing; set BREPAIR_EXECUTABLE or install a built wheel')
    command = [str(selected), *map(str, arguments)]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'BRe-Pair exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}')
    return RunResult(tuple(command), result.stdout, result.stderr, time.perf_counter()-start)


def train_file(source, archive, *, minimum_count=256, threads=4, memory='2GB',
               score='weighted-pmi', fill_waves=2, maximum_rules=None, executable=None):
    args = ['train', source, archive, '--min-count', minimum_count, '--threads', threads,
            '--memory', memory, '--score', score, '--fill-waves', fill_waves]
    if maximum_rules is not None:
        args += ['--max-rules', maximum_rules]
    return _run(args, executable)


def select_vocabulary(archive, training_source, model, *, vocabulary_size=32768,
                      minimum_uses=8, threads=4, memory='2GB', executable=None):
    """Use the supported CLI statistics selector; research workbench arms are separate."""
    return _run(['vocab', archive, training_source, model, '--vocab-size', vocabulary_size,
                 '--min-token-uses', minimum_uses, '--selection', 'statistics', '--threads', threads,
                 '--memory', memory], executable)


def compress_file(source, archive, *, minimum_count=2, threads=4, memory='2GB',
                  score='weighted-pmi', fields='lines', reparse_refits=0, executable=None):
    return _run(['compress', source, archive, '--min-count', minimum_count, '--threads', threads,
                 '--memory', memory, '--score', score, '--fields', fields,
                 '--reparse-refits', reparse_refits], executable)


def decompress_file(archive, output, *, threads=1, memory='2GB', executable=None):
    with Path(archive).open('rb') as stream:
        magic = stream.read(8)
    if magic == b'BRPAIR01':
        return _run(['decompress', archive, output, '--memory', memory], executable)
    return _run(['huffman-decode', archive, output, '--threads', threads, '--memory', memory], executable)
