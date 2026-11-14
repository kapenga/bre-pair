"""Independent lexical column archives for unquoted pipe-separated tables."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ._commands import _run, compress_file, decompress_file


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup(work, parent):
    # Only recursively remove the private directory created by this operation.
    work, parent = Path(work).resolve(), Path(parent).resolve()
    if work.parent != parent or not work.name.startswith('.brepair-columns-'):
        raise RuntimeError('Unexpected column scratch directory')
    shutil.rmtree(work)


def compress_columns(source, directory, *, columns, minimum_count=2, threads=4,
                     memory='2GB', score='weighted-pmi', fields='lines', executable=None,
                     progress=None):
    """Train/store one complete BRe-Pair archive per lexical column, serially.

    Quotes are literal and pipes/LF are separators; this is not quoted CSV or
    typed Parquet. fields='lines' gives value EOF; 'single' is a bulk-column mode.
    """
    source, root = Path(source).resolve(), Path(directory).resolve()
    root.mkdir(exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='.brepair-columns-', dir=root))
    try:
        original_sha = _sha(source)
        extracted = work/'values'
        result = _run(['columns', 'split', source, extracted, columns], executable)
        layout = json.loads(result.stdout.strip())
        layout_bytes = (extracted/'layout.brcol').read_bytes()
        (root/'layout.brcol').write_bytes(layout_bytes)
        archives = []
        for column in range(layout['columns']):
            name = f'c{column:02}.brh'
            run = compress_file(extracted/f'c{column:02}.txt', root/name,
                minimum_count=minimum_count, threads=threads, memory=memory, score=score,
                fields=fields, executable=executable)
            archives.append(dict(column=column, file=name, bytes=(root/name).stat().st_size,
                                 sha256=_sha(root/name), construction_seconds=run.seconds))
            if progress is not None:
                progress(column+1, layout['columns'])
        if _sha(source) != original_sha:
            raise RuntimeError('Source changed during column construction')
        manifest = dict(format='BRCOLARCH01', layout=layout, layout_sha256=_sha(root/'layout.brcol'),
                        source_sha256=original_sha, fields=fields, score=score,
                        minimum_count=minimum_count, archives=archives,
                        contract='unquoted PSV; quotes/null spellings literal; exact row/byte restoration')
        with (root/'manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
        return dict(directory=str(root), columns=layout['columns'], rows=layout['rows'],
                    complete_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file()))
    finally:
        _cleanup(work, root)


def _manifest(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'BRCOLARCH01':
        raise ValueError('Not a BRe-Pair column archive')
    columns = manifest['layout']['columns']
    if not isinstance(columns, int) or not 1 <= columns <= 1024 or len(manifest['archives']) != columns:
        raise ValueError('Invalid column count')
    for column, archive in enumerate(manifest['archives']):
        if archive['column'] != column or archive['file'] != f'c{column:02}.brh':
            raise ValueError('Invalid column archive mapping')
    if _sha(root/'layout.brcol') != manifest['layout_sha256']:
        raise ValueError('Column layout changed')
    return root, manifest


def decompress_column(directory, column, output, *, threads=1, memory='2GB', executable=None):
    """Decode one projected column; values are LF-delimited lexical bytes."""
    root, manifest = _manifest(directory)
    if not isinstance(column, int) or not 0 <= column < manifest['layout']['columns']:
        raise ValueError('Column ordinal outside archive')
    return decompress_file(root/manifest['archives'][column]['file'], output,
                           threads=threads, memory=memory, executable=executable)


def decompress_columns(directory, output, *, threads=1, memory='2GB', executable=None):
    """Restore exact original PSV bytes; verify SHA before publishing output."""
    root, manifest = _manifest(directory)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    work = Path(tempfile.mkdtemp(prefix='.brepair-columns-', dir=output.parent))
    try:
        (work/'layout.brcol').write_bytes((root/'layout.brcol').read_bytes())
        for column, archive in enumerate(manifest['archives']):
            decompress_file(root/archive['file'], work/f'c{column:02}.txt', threads=threads,
                            memory=memory, executable=executable)
        restored = work/'restored.psv'
        _run(['columns', 'restore', work, restored], executable)
        if restored.stat().st_size != manifest['layout']['input_bytes'] or _sha(restored) != manifest['source_sha256']:
            raise ValueError('Restored table differs from original SHA-256')
        # Same-parent scratch guarantees the same filesystem; hard-link creation
        # is atomic and refuses to overwrite a concurrently created destination.
        os.link(restored, output)
        return output
    finally:
        _cleanup(work, output.parent)
