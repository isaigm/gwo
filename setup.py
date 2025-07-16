# setup.py
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Una clase de extensión de CMake que sabe cómo ejecutar CMake
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

# Una clase de construcción personalizada que ejecuta CMake
class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # El directorio donde se colocarán los archivos de compilación de CMake
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Configuración de CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={Path(self.get_ext_fullpath(ext.name)).parent.resolve()}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",  # Asegúrate de construir en modo Release para optimizaciones
        ]
        
        # Comando de configuración de CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        
        # Comando de compilación de CMake
        subprocess.check_call(
            ["cmake", "--build", "."], cwd=build_temp
        )

# La configuración principal del paquete
setup(
    name="gwo_py",
    version="0.1.0",
    author="Isai",
    author_email="isaigm23@gmail.com",
    description="Una implementación de GWO en C++ con binding de Python",
    long_description="",
    # Aquí le decimos a setuptools que nuestro módulo es una extensión de CMake
    ext_modules=[CMakeExtension("gwo_py")],
    # Aquí le decimos que use nuestra clase de construcción personalizada
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)