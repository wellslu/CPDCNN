# # from setuptools import setup
# # from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# # import os

# # # Define the CUDA source files relative to the setup.py location in `lib`
# # source_files = ["util.cu"]

# # # Set up the build output directory
# # build_dir = "../build"  # Modify this to place the build files in the desired directory

# # setup(
# #     name="util",
# #     ext_modules=[
# #         CUDAExtension(
# #             "util",              # The module name
# #             sources=source_files,  # CUDA source file(s)
# #             extra_compile_args={'cxx': ['-DBUILD_WITH_PYTORCH']}  # Define macro
# #         )
# #     ],
# #     cmdclass={"build_ext": BuildExtension},
# #     options={
# #         "build": {"build_base": build_dir}  # Specify the base directory for the build output
# #     },
# # )

# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# import os

# # Define the CUDA source files relative to setup.py location
# source_files = ["util.cu"]  # Adjusted path to match `lib` directory

# # Set up the build output directory
# build_dir = "../build"  # Modify if you want a different output location

# setup(
#     name="util",
#     ext_modules=[
#         CUDAExtension(
#             "util",               # The module name
#             sources=source_files, # CUDA source files
#             extra_compile_args={'cxx': ['-DBUILD_WITH_PYTORCH']},  # Define macro
#         )
#     ],
#     cmdclass={"build_ext": BuildExtension},
#     # Set the build directory for build output
#     script_args=["build_ext", f"--build-lib={build_dir}"],
# )
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the CUDA source files
source_files = ["util.cu"]

# Set up the build output directory
build_dir = "../build/lib"

setup(
    name="util",
    ext_modules=[
        CUDAExtension(
            name="util",
            sources=source_files,
            extra_compile_args={
                'cxx': ['-DBUILD_WITH_PYTORCH'],
                'nvcc': ['-DBUILD_WITH_PYTORCH']  # Define macro for nvcc
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext", f"--build-lib={build_dir}"],
)