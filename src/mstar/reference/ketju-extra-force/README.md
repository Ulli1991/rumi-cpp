# Ketju-integrator (MSTAR)

A highly parallel regularized N-body integrator library.

## Building
Depending on whether you want to build the included unit tests 
(recommended for developing the code), you will either first need to run
    
    git submodule update --init

to pull in the unit test libraries required for building the tests;
or pass `-DUNIT_TESTS=OFF` as an option to `cmake` in the commands below
to disable the tests.

Run

    mkdir build
    cd build
    cmake .. [CMAKE_OPTIONS]

to configure the build system.
Configuration options can then be changed by editing the `CMakeCache.txt` file
either directly or using e.g. `ccmake`, but doing this is not typically required.

After configuration, the library, a python shim and the standalone driver 
program can be built by running `make` in the build directory.
The standalone `nbody` program is found under `bin/`.
*TODO: improve and document usage of `nbody`.*

## Usage in other codes

The public interface of the library is included in the file 
`include/ketju_integrator/ketju_integrator.h`,
so `#include <ketju_integrator/ketju_integrator.h>` and add
`-I/path/to/include/` to your compiler flags.
To link the library, add `-L/path/to/lib -lketju-integrator` to your linker
flags.
*TODO: usage in cmake projects*

## Python bindings

Simple Python 3 bindings depending only on the standard library and NumPy are
included as the `ketju_integrator` module under `python_bindings`.
They function via a small C library shim and allow parallel integration by
running the integrator in subprocesses with MPI.
An example of using the bindings is in
`python_bindings/ketju_integrator_example.py`.
For more convenient usage add the `python_bindings` directory to the `PYTHONPATH`.
