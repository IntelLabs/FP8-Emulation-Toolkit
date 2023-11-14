import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

cmdclass = {}
ext_modules = []

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules.append(
        CppExtension('fpemu_cpp',
            ['mpemu/pytquant/cpp/fpemu_impl.cpp'],
            extra_compile_args = ["-mf16c", "-march=native", "-mlzcnt", "-fopenmp", "-Wdeprecated-declarations"]
        ),)

if torch.cuda.is_available():
   from torch.utils.cpp_extension import BuildExtension, CUDAExtension
   ext_modules.append(
        CUDAExtension('fpemu_cuda', [
            'mpemu/pytquant/cuda/fpemu_impl.cpp',
            'mpemu/pytquant/cuda/fpemu_kernels.cu'],
        ),)

ext_modules.append(
    CppExtension('simple_gemm_dev', 
        ['mpemu/cmodel/simple/simple_gemm.cpp', 'mpemu/cmodel/simple/simple_gemm_impl.cpp', 'mpemu/cmodel/simple/simple_mm_engine.cpp'], 
        extra_compile_args=[ '-march=native', '-fopenmp','-Wunused-but-set-variable','-Wunused-variable'],
        include_dirs=['{}/'.format(os.getenv("PWD")+'/mpemu/cmodel/simple')],
    ),)

ext_modules.append(
    CppExtension('simple_conv2d_dev', 
        ['mpemu/cmodel/simple/simple_conv2d.cpp', 'mpemu/cmodel/simple/simple_conv2d_impl.cpp', 'mpemu/cmodel/simple/simple_mm_engine.cpp'], 
        extra_compile_args=[ '-march=native', '-fopenmp','-Wunused-but-set-variable','-Wunused-variable'],
        include_dirs=['{}/'.format(os.getenv("PWD")+'/mpemu/cmodel/simple')],
    ),)

cmdclass['build_ext'] = BuildExtension

setup(
    name="mpemu",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    author="Naveen Mellempudi",
    description="FP8 Emulation Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
