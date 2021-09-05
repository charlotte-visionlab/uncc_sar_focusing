# UNCC SAR Focusing 

A codebase for developing SAR focusing algorithms that can be run from MATLAB or from the command line on the host.

# Library dependencies

This code compiles to (4) different targets:

(1) a CPU-only application which depends on the libfftw3-dev package (libfftw),
(2) a CPU-only MATLAB application which includes the dependencies of (1) and also depends on MATLAB being installed,
(3) a GPU application which depends on CUDA, and cuFFT.
(4) a GPU MATLAB application which includes the dependencies of (4) and also depends on MATLAB (2017 or later) being installed.

One must also have installed libhdf5-dev to run this code.

# Before you build define the CUDA and MATLAB home folders if they are available

MATLAB and CUDA can be installed to custom locations. Make sure to edit CMakeLists.txt in the root folder to specify their locations.
MATLAB's location is specified by modifying the line in this file below:
```
set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2021a")
```
CUDA's location is specified by modifying the line in this file below:
```
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
```

# Update the matio library
Before compiling you will want to download the source code to matio via the command below:
```
git submodule update --init
```
For linux builds you will need to modify the matio build. Specifically one must comment out line 71 of the matio/cmake/thirdParties.cmake by putting a '#' symbol at the beginning of the line as shown below:
```
#target_link_libraries(MATIO::ZLIB INTERFACE ${target})
```
# CMake-style compile
To compile the code from the root folder of the project execute the following shell commands:
```
mkdir build
cd build
cmake ..
make
```
Application targets will be compiled into the build/bin/... subfolder.
