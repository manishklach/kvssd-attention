import os

from setuptools import setup


ext_modules = []
cmdclass = {}
if os.getenv("KVSSD_BUILD_CUDA") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            "kvssd._C",
            ["csrc/bindings.cpp", "csrc/fused_decode.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)

