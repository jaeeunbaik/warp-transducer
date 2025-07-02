from distutils.version import LooseVersion
import os
import platform
import sys
from setuptools import setup, find_packages
import torch
# CUDA 확장을 위해 CUDAExtension을 임포트합니다.
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, include_paths, library_paths

# Conda 환경 경로를 가져와 환경 변수로 설정합니다.
# 이 부분은 setup.py가 실행될 때 빌드 환경에 정확한 경로를 제공하도록 합니다.
conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
os.environ['CUDA_HOME'] = conda_env_path
os.environ['LD_LIBRARY_PATH'] = os.path.join(conda_env_path, 'lib') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDACXX'] = os.path.join(conda_env_path, 'bin', 'nvcc')
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CUDAHOSTCC'] = '/usr/bin/g++'

# 컴파일 인자 설정
# nvcc에는 '-fPIC'를 직접 전달하지 않습니다. g++(cxx)에만 전달합니다.
extra_compile_args = {'cxx': ['-fPIC'], 'nvcc': []}
if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    extra_compile_args['cxx'] += ['-std=c++17']
    extra_compile_args['nvcc'] += ['-std=c++17'] # PyTorch 1.5.0+는 C++17을 요구하므로 nvcc에도 전달
else:
    # PyTorch 1.1.0을 사용한다면 이 부분이 C++14로 설정될 것입니다.
    # 이전 오류에서 C++17을 요구하는 PyTorch 헤더가 나왔으므로,
    # PyTorch 1.1.0을 사용한다면 이 부분이 C++14로 설정될 것입니다.
    extra_compile_args['cxx'] += ['-std=c++14']
    extra_compile_args['nvcc'] += ['-std=c++14'] # nvcc에도 C++14 전달

warp_rnnt_path = "../build"

# CUDA 지원 여부를 확인합니다.
# CUDA_HOME 환경 변수가 설정되어 있으면 GPU 확장을 빌드합니다.
if torch.cuda.is_available() or "CUDA_HOME" in os.environ:
    enable_gpu = True
else:
    print("Torch was not built with CUDA support, not building GPU extensions.")
    enable_gpu = False

# 운영체제에 따라 라이브러리 확장자를 설정합니다.
if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

# GPU 확장이 활성화되면 컴파일 인자에 GPU 관련 매크로를 추가합니다.
if enable_gpu:
    # 이 매크로는 binding.cpp에서 #ifdef WARPRNNT_ENABLE_GPU 로 사용됩니다.
    extra_compile_args['cxx'] += ['-DWARPRNNT_ENABLE_GPU']
    extra_compile_args['nvcc'] += ['-DWARPRNNT_ENABLE_GPU']

# 핵심 C++ 라이브러리 (libwarprnnt.so)의 존재 여부를 확인합니다.
# 이 라이브러리는 warp-transducer 프로젝트의 루트에서 먼저 빌드되어야 합니다.
if "WARP_RNNT_PATH" in os.environ:
    warp_rnnt_path = os.environ["WARP_RNNT_PATH"]
if not os.path.exists(os.path.join(warp_rnnt_path, "libwarprnnt" + lib_ext)):
    print(("Could not find libwarprnnt{} in {}.\n"
           "Build warp-rnnt (the C++ library) and set WARP_RNNT_PATH to the location of"
           " libwarprnnt{} (default is '../build')").format(lib_ext, warp_rnnt_path, lib_ext))
    sys.exit(1)

# 컴파일러가 헤더 파일을 찾을 경로를 설정합니다.
# PyTorch의 기본 include 경로와 CUDA 툴킷의 include 경로를 명시적으로 추가합니다.
include_dirs = [
    os.path.realpath('../include'), # warp-transducer 자체의 헤더
    os.path.join(torch.utils.cpp_extension.include_paths()[0]), # PyTorch의 기본 C++ 헤더 경로
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'TH'), # PyTorch의 구형 CUDA 관련 헤더 (THC.h가 있는 경우)
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'THC'), # PyTorch의 구형 CUDA 관련 헤더 (THC.h가 있는 경우)
    os.path.join(conda_env_path, 'include'), # CUDA 툴킷의 핵심 헤더 (cuda_runtime_api.h 등이 있는 곳)
    os.path.join(conda_env_path, 'targets', platform.machine() + '-linux', 'include') # Conda CUDA의 targets/x86_64-linux/include 경로
]

# setup 함수 호출
setup(
    name='warprnnt_pytorch',
    version="0.1",
    description="PyTorch wrapper for RNN-Transducer",
    url="https://github.com/HawkAaron/warp-transducer",
    author="Mingkun Huang",
    author_email="mingkunhuang95@gmail.com",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension( # GPU 확장을 위해 CUDAExtension을 사용합니다.
            name='warprnnt_pytorch.warp_rnnt',
            sources=['src/binding.cpp'], # src/cuda_kernels.cu 파일을 제거했습니다.
            include_dirs=include_dirs,
            library_dirs=[os.path.realpath(warp_rnnt_path)], # libwarprnnt.so가 있는 경로
            libraries=['warprnnt'], # 링크할 라이브러리 이름
            extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_rnnt_path)], # 런타임 라이브러리 경로
            extra_compile_args=extra_compile_args # 컴파일 인자
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
