import os

from setuptools import setup


ext_modules = []
cmdclass = {}
if os.getenv("KVSSD_BUILD_CUDA") == "1" or os.getenv("KVSSD_BUILD_GDS") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if os.getenv("KVSSD_BUILD_CUDA") == "1":
        ext_modules.append(
            CUDAExtension(
                "kvssd._C",
                ["csrc/bindings.cpp", "csrc/fused_decode.cu"],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
                },
            )
        )
    if os.getenv("KVSSD_BUILD_GDS") == "1":
        ext_modules.append(
            CUDAExtension(
                "kvssd._gds",
                ["csrc/gds_bindings.cpp"],
                libraries=["cufile"],
                extra_compile_args={"cxx": ["-O3"]},
            )
        )
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
