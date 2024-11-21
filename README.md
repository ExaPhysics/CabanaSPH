# CabanaSPH

Smoothed particle hydrodynamics in Cabana.

## Dependencies
CabanaDEM has the following dependencies:

|Dependency | Version  | Required | Details|
|---------- | -------  |--------  |------- |
|CMake      | 3.11+    | Yes      | Build system
|Cabana     | a73697f  | Yes      | Performance portable particle algorithms

Cabana must be built with the following in order to work with CabanaPD:
|Cabana Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.16+   | Yes      | Build system
|Kokkos     | 3.7.0+  | Yes      | Performance portable on-node parallelism
|HDF5       | master  | Yes       | Particle output
|SILO       | master  | No       | Particle output

The underlying parallel programming models are available on most systems, as is
CMake. Those must be installed first, if not available. Kokkos and Cabana are
available on some systems or can be installed with `spack` (see
https://spack.readthedocs.io/en/latest/getting_started.html):

```
spack install cabana@master+cajita+silo
```

Alternatively, Kokkos can be built locally, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana must be
compiled with MPI and the Grid sub-package.

## Obtaining CabanaDEM

Clone the master branch:

```
git clone https://github.com/ExaPhysics/CabanaDEM.git
```

## Build and install
### CPU Build

After building Kokkos and Cabana for CPU, the following script will build and install CabanaDEM:

```
#Change directory as needed
export CABANA_INSTALL_DIR=/home/username/Cabana/build/install

cd ./CabanaDEM
mkdir build
cd build
cmake \
    -D CMAKE_PREFIX_PATH="$CABANA_INSTALL_DIR" \
    -D CMAKE_INSTALL_PREFIX=install \
    .. ;
make install
```

### CUDA Build

After building Kokkos and Cabana for Cuda:
https://github.com/ECP-copa/Cabana/wiki/CUDA-Build

The CUDA build script is identical to that above, but again note that Kokkos
must be compiled with the CUDA backend.

Note that the same compiler should be used for Kokkos, Cabana, and CabanaPD.

### HIP Build

After building Kokkos and Cabana for HIP:
https://github.com/ECP-copa/Cabana/wiki/HIP-and-SYCL-Build#HIP

The HIP build script is identical to that above, except that `hipcc` compiler
must be used:

```
-D CMAKE_CXX_COMPILER=hipcc
```

Note that `hipcc` should be used for Kokkos, Cabana, and CabanaPD.


### More detailed Cabana and Kokkos installation

A more detailed installation can also be found at:
https://gist.github.com/dineshadepu/4313d6a148f7188965b951406e9fb83f
