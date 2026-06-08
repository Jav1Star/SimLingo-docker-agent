from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="box overlaps",
    ext_modules=cythonize(
        [Extension("box_overlaps", [str(ROOT / "box_overlaps.pyx")])],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
)