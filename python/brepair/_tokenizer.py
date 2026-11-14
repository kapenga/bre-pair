import ctypes as C
from functools import lru_cache
import operator
import os
from pathlib import Path
import sys
from threading import RLock


class _Bytes(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_uint8)), ('length', C.c_size_t)]


class _Ids(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_uint32)), ('length', C.c_size_t)]


@lru_cache(maxsize=1)
def _library():
    name = 'bre_pair_native.dll' if os.name == 'nt' else ('libbre_pair_native.dylib' if sys.platform == 'darwin' else 'libbre_pair_native.so')
    path = Path(os.environ.get('BREPAIR_NATIVE_LIBRARY', Path(__file__).parent/'_native'/name))
    if not path.is_file():
        raise RuntimeError('Native BRe-Pair library missing. Install a built wheel, or build with pip install . from the repository root.')
    lib = C.CDLL(str(path.resolve()))  # ctypes releases the GIL during native calls.
    signatures = {
        'brepair_abi_version': (C.c_uint32, []),
        'brepair_last_error': (C.c_char_p, []),
        'brepair_tokenizer_load': (C.c_void_p, [C.c_char_p, C.c_size_t]),
        'brepair_tokenizer_from_bytes': (C.c_void_p, [C.c_char_p, C.c_size_t]),
        'brepair_tokenizer_fork': (C.c_void_p, [C.c_void_p]),
        'brepair_tokenizer_free': (None, [C.c_void_p]),
        'brepair_tokenizer_size': (C.c_size_t, [C.c_void_p]),
        'brepair_tokenizer_bytes': (C.c_int32, [C.c_void_p, C.POINTER(_Bytes)]),
        'brepair_tokenizer_encode': (C.c_int32, [C.c_void_p, C.c_char_p, C.c_size_t, C.POINTER(_Ids)]),
        'brepair_tokenizer_decode': (C.c_int32, [C.c_void_p, C.POINTER(C.c_uint32), C.c_size_t, C.c_size_t, C.POINTER(_Bytes)]),
        'brepair_bytes_free': (None, [_Bytes]),
        'brepair_ids_free': (None, [_Ids]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    if lib.brepair_abi_version() != 1:
        raise RuntimeError('Unsupported BRe-Pair native ABI')
    return lib


def _error(lib):
    return ValueError((lib.brepair_last_error() or b'Native BRe-Pair error').decode('utf-8', errors='replace'))


class Tokenizer:
    """Immutable BRTOKN01 model with an exact native parser and reusable workspace.

    Each encode call represents one complete field and includes its EOF, which
    may be fused into a learned token. Bytes remain unchanged; str uses UTF-8.
    Use fork() for independent worker workspaces sharing the immutable model.
    """
    def __init__(self, model_path):
        self._lib = _library()
        self._lock = RLock()
        self._handle = None
        path = str(Path(model_path)).encode('utf-8')
        self._handle = self._lib.brepair_tokenizer_load(path, len(path))
        if not self._handle:
            raise _error(self._lib)

    @classmethod
    def _from_handle(cls, lib, handle):
        if not handle:
            raise _error(lib)
        obj = cls.__new__(cls)
        obj._lib, obj._lock, obj._handle = lib, RLock(), handle
        return obj

    @classmethod
    def from_bytes(cls, model_bytes):
        data = bytes(model_bytes)
        lib = _library()
        return cls._from_handle(lib, lib.brepair_tokenizer_from_bytes(data, len(data)))

    def _open(self):
        if not self._handle:
            raise RuntimeError('Tokenizer is closed')
        return self._handle

    def fork(self):
        with self._lock:
            return self._from_handle(self._lib, self._lib.brepair_tokenizer_fork(self._open()))

    @property
    def vocabulary_size(self):
        with self._lock:
            return self._lib.brepair_tokenizer_size(self._open())

    def encode(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        else:
            raise TypeError('encode expects str or a bytes-like object')
        with self._lock:
            output = _Ids()
            if self._lib.brepair_tokenizer_encode(self._open(), data, len(data), C.byref(output)):
                raise _error(self._lib)
            try:
                return output.data[:output.length]
            finally:
                self._lib.brepair_ids_free(output)

    def encode_fields(self, fields):
        """Yield one ID list per field without materializing the whole corpus."""
        for field in fields:
            yield self.encode(field)

    def decode(self, ids, *, maximum_bytes=64 << 20):
        values = [operator.index(value) for value in ids]
        if any(value < 0 or value > 0xffffffff for value in values):
            raise ValueError('Token IDs must fit unsigned 32-bit integers')
        maximum = operator.index(maximum_bytes)
        if maximum < 0 or maximum > C.c_size_t(-1).value:
            raise ValueError('maximum_bytes must fit an unsigned native size')
        source = (C.c_uint32*len(values))(*values)
        with self._lock:
            output = _Bytes()
            if self._lib.brepair_tokenizer_decode(self._open(), source, len(values), maximum, C.byref(output)):
                raise _error(self._lib)
            try:
                return C.string_at(output.data, output.length)
            finally:
                self._lib.brepair_bytes_free(output)

    def to_bytes(self):
        with self._lock:
            output = _Bytes()
            if self._lib.brepair_tokenizer_bytes(self._open(), C.byref(output)):
                raise _error(self._lib)
            try:
                return C.string_at(output.data, output.length)
            finally:
                self._lib.brepair_bytes_free(output)

    def save(self, path):
        data = self.to_bytes()
        with Path(path).open('xb') as stream:
            stream.write(data)

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.brepair_tokenizer_free(self._handle)
                self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if getattr(self, '_handle', None):
            self.close()
