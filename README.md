# Collection Software for Event & RGBD Camera
This is a software for collecting data from event camera and RGBD camera.
We build the hardware with Prophesee EVK4 Event camera together with ORBBEC's Femote Bolt.

To install the software, please put ORBBEC-SDK in the thirdparty folder and install the Prophesee's metavision SDK of 4.6 version yourself.

The software is built with CMake. Any dependencies should be checked carefully before building.

The software doesn't support hardware re-initialization under one run. If the hardware trigger stopped, the system should also be re-run for correct initialization.

For now, the system proceed correctly under Ubuntu 20.04, if you encounter any corner cases, please contact me to fix it.

## Usage
```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./scanner_cpp_optim --store /path/to/store (or none which causes the storage folder default to build folder)
```

## Declaration
This software is developed by haoming-Yu at Zhejiang University. The code is released under the MIT license.