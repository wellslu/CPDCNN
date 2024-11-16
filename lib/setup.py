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
import os
# Recursively collect all .cu files under the current directory (`lib`) and subdirectories
def collect_source_files(directory, extension=".cu"):
    sources = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                sources.append(os.path.join(root, file))
    return sources

# Define directories
source_dir = "."  # Current directory (relative to where `setup.py` is located)
# Define the CUDA source files
source_files = collect_source_files(source_dir)

# Get all subdirectories under 'inc'
def get_include_dirs(base_dir):
    include_dirs = [base_dir]
    for root, dirs, _ in os.walk(base_dir):
        for sub_dir in dirs:
            include_dirs.append(os.path.join(root, sub_dir))
    return include_dirs

# Base directory for includes
base_include_dir = os.path.abspath('../inc')

# Set up the build output directory
temp_dir = "../build" # for temp file obj file
build_dir = "../build/lib" # for python import place

setup(
    name="util",
    ext_modules=[
        CUDAExtension(
            name="util",
            sources=source_files,
            include_dirs=get_include_dirs(base_include_dir),
            extra_compile_args={
                'cxx': ['-DBUILD_WITH_PYTORCH'],
                'nvcc': ['-DBUILD_WITH_PYTORCH']  # Define macro for nvcc
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={
        "build": {"build_base": temp_dir}  # Specify the base directory for the build output
    },
    script_args=["build_ext", f"--build-lib={build_dir}"],
)