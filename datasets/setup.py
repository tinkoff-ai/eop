
import os

from setuptools import setup, Extension

os.environ['CFLAGS'] = '-std=c++11'

if __name__ == "__main__":
    from numpy import get_include
    from Cython.Build import cythonize

    # setup Cython build
    ext = Extension('dataset',
                    sources=['dataset.pyx'],
                    include_dirs=[get_include(), 'include'],
                    language='c++',
                    extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
                    extra_link_args=["-std=c++11"])

    ext_modules = cythonize([ext],
                            compiler_directives={
                                'linetrace': True,
                                'binding': True
                            })

    setup(name="rl-datasets", ext_modules=ext_modules)