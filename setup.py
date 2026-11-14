"""Build the native library and existing executable into a platform wheel."""
import os
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        return 'py3', 'none', super().get_tag()[2]


class BuildNative(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        target = root/'build/cargo-python'
        cargo = os.environ.get('CARGO') or shutil.which('cargo')
        if cargo is None:
            candidate = Path.home()/'.cargo/bin'/('cargo.exe' if os.name == 'nt' else 'cargo')
            if candidate.is_file():
                cargo = str(candidate)
        if cargo is None:
            raise RuntimeError('Rust cargo 1.87+ is required to build BRe-Pair from source')
        for command in [
            [cargo, 'build', '--locked', '--profile', 'dist', '--manifest-path', 'bindings/python-native/Cargo.toml', '--target-dir', str(target)],
            [cargo, 'build', '--locked', '--profile', 'dist', '--bin', 'brepair', '--target-dir', str(target)],
        ]:
            subprocess.run(command, cwd=root, check=True)
        destination = Path(self.build_lib)/'brepair/_native'
        destination.mkdir(parents=True, exist_ok=True)
        library = 'bre_pair_native.dll' if os.name == 'nt' else ('libbre_pair_native.dylib' if sys.platform == 'darwin' else 'libbre_pair_native.so')
        for name in [library, 'brepair.exe' if os.name == 'nt' else 'brepair']:
            shutil.copy2(target/'dist'/name, destination/name)
        licenses = destination/'licenses'
        licenses.mkdir(exist_ok=True)
        for path in [root/'LICENSE', root/'NOTICE', root/'THIRD-PARTY-NOTICES.md', *root.glob('third-party/*.txt')]:
            shutil.copy2(path, licenses/path.name)
        source = [root/'Cargo.toml', root/'Cargo.lock', *root.glob('src/**/*.rs'),
                  root/'bindings/python-native/Cargo.toml',root/'bindings/python-native/Cargo.lock',
                  *root.glob('.cargo/*.toml'),*root.glob('bindings/python-native/src/*.rs')]
        digest = hashlib.sha256()
        for path in sorted(source, key=lambda p: p.relative_to(root).as_posix()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b'\0')
            digest.update(path.read_bytes())
        rustc = os.environ.get('RUSTC') or shutil.which('rustc') or str(Path(cargo).with_name('rustc.exe' if os.name=='nt' else 'rustc'))
        info = dict(native_abi=1, core_version='0.1.0', platform=platform.platform(),
                    rustc=subprocess.check_output([rustc,'--version'],cwd=root,text=True).strip(),
                    cargo=subprocess.check_output([cargo,'--version'],cwd=root,text=True).strip(),
                    source_sha256=digest.hexdigest(), automatic_avx2_minimum_symbols=1048576)
        (destination/'build-info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')


setup(distclass=NativeDistribution, cmdclass={'build_py':BuildNative, 'bdist_wheel':PlatformWheel})
