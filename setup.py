from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
root_dir = '/usr/include/'
extensions = [
    Extension("sdp_admm_py", ["sdp_admm_py.pyx", "sdp_admm.cpp"],
              include_dirs=[root_dir + 'eigen3']
            )
    ]
with open("README.md") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setup(
        name = 'sdp_admm_py',
        ext_modules = cythonize(extensions),
        url = 'https://github.com/zhaofeng-shu33/sdp_admm_py',
        version = '0.3',
        author = 'zhaofeng-shu33',
        author_email = '616545598@qq.com',
        long_description = long_description,
        long_description_content_type="text/markdown",           
        license = 'Apache License Version 2.0',
        description = 'SBM community detection with semi-definite programming'
    )


