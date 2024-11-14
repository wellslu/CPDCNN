# CPDCNN

## Quick Start

Check the tool version first 
```
nvcc 11.7 (C++ 11.4)
pytorch 2.0.1 (compatible with nvcc 11.7)
python 3.10.12 (if not, revise including dir path in CMakeLists.txt)
```
To set up the environment, run:

```bash
git submodule update --init --recursive
source ./configure
```
After source the script, you will locate at build directory.
The executable files(low-level cuda implementation) can be found in build/bin
Python scripts will be in build/app, to execute those python file please cd to the current directory place.