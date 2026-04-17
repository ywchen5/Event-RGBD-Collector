# Collection Software for Event & RGBD Camera
This is a software for collecting data from event camera and RGBD camera.
We build the hardware with Prophesee EVK4 Event camera together with ORBBEC's Femote Bolt.

To install the software, please put ORBBEC-SDK in the thirdparty folder and install Prophesee's Metavision SDK 4.6 yourself.

Windows users must also register Orbbec metadata before running the collector. See `thirdparty/OrbbecSDK/shared/obsensor_metadata_win10.md` and run `thirdparty/OrbbecSDK/shared/obsensor_metadata_win10.ps1 -op install_all` from an elevated PowerShell after connecting the device.

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

```
Set-Location D:\ywchen\Event-RGBD-Collector\build\Release
$env:PATH="D:\ywchen\Event-RGBD-Collector\thirdparty\OrbbecSDK\bin;C:\Program Files\Prophesee\bin;C:\Program Files\Prophesee\third_party\bin;C:\Windows\System32;C:\Windows"
.\scanner_cpp_optim.exe --store --output D:\ywchen\data_out
```

If the program reports `Orbbec camera initialization failed: No device found`, first verify that Windows Device Manager sees the camera, then rerun the metadata registration script above. This error usually means the SDK cannot enumerate a connected device, not that the RGBD pipeline logic is broken.