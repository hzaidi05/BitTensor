from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="bittensor",
      ext_modules=[
          cpp_extension.CppExtension(
            "bittensor",
            ["src/torch/bindings.cpp"],
            extra_compile_args={
                "cxx": [
                    "-std=c++17",  # Enable C++17
                    "-O3",         # Enable optimizations
                    "-march=native",  # Enable CPU-specific optimizations
                    "-ffast-math",    # Enable fast math optimizations
                    "-DPy_LIMITED_API=0x03090000"  # Python limited API
                ]
            },
            py_limited_api=True)],  # Build 1 wheel across multiple Python versions
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
)