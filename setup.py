from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        'sparsehess',
        ['sparsehess.pyx'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
