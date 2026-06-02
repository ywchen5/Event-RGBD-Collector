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

To test a visual RGB/event timing offset without changing the raw timestamp
diagnostics, pass either a microsecond value or a frame count:

```powershell
.\scanner_cpp_optim.exe --store --output D:\ywchen\data_out --rgb-event-offset-frames 2
.\scanner_cpp_optim.exe --store --output D:\ywchen\data_out --rgb-event-offset-us 66666
```

For RGB timing diagnostics, the collector logs Orbbec color exposure-related
device properties at startup and can force manual color settings:

```powershell
Set-Location D:\ywchen\Event-RGBD-Collector\build\Release
$env:PATH="D:\ywchen\Event-RGBD-Collector\thirdparty\OrbbecSDK\bin;C:\Program Files\Prophesee\bin;C:\Program Files\Prophesee\third_party\bin;C:\Windows\System32;C:\Windows"
.\scanner_cpp_optim.exe --store --output D:\ywchen\data_out `
  --color-auto-exposure 0 --color-exposure 100 --color-gain 32 `
  --color-auto-white-balance 0 --color-white-balance 4600
```

Useful options:

- `--color-auto-exposure 0|1`
- `--color-exposure N`
- `--color-gain N`
- `--color-auto-white-balance 0|1`
- `--color-white-balance N`
- `--color-auto-exposure-priority N`
- `--color-power-line-frequency N`
- `--color-format MJPG|BGR|RGB|YUYV|UYVY|NV12`

Use the startup log ranges and the read-back snapshot as the source of truth for
valid exposure/gain values on the connected Femto Bolt. On the current Femto
Bolt/SDK, `OB_PROP_COLOR_EXPOSURE_INT` reports `min=1 max=300 step=1`; this is
consistent with the SDK metadata convention that color exposure is usually in
100 us units, so `--color-exposure 100` is approximately 10 ms. Do not pass
microseconds directly: `--color-exposure 8000` is outside this range and reads
back as the previous/default value. A successful manual exposure setup should
log and later record:

```text
Color property snapshot after config: auto_exposure=0 exposure=100 ...
```

If a non-MJPG color format is unsupported, the SDK will report that during
initialization; use `MJPG` as the baseline.

`frame\sync_timestamps.csv` also records trigger/frame sequence diagnostics:

- `event_trigger_start_seq`, `event_trigger_end_seq`: accepted Prophesee trigger sequence for each event slice.
- `rgb_frame_index`, `depth_frame_index`: Orbbec aligned frame indices.
- `raw_rgb_frame_index`, `raw_depth_frame_index`: Orbbec raw frame indices before alignment.
- `rgb_color_format`: actual Orbbec color frame format enum value.
- `rgb_prop_*`: Orbbec color device property snapshot read near each saved frame, including exposure and gain.
- `*_system_ts_us`: Orbbec SDK system timestamps for arrival/output timing checks.
- `*_global_ts_us`: Orbbec SDK global timestamps, when available.
- `*_metadata_size`: raw metadata buffer size reported by the SDK.
- `*_metadata_hex`: raw metadata bytes written as a continuous hex string.
- `*_meta_sensor_timestamp`: Orbbec frame metadata timestamp reported as the middle of capture by the SDK.
- `*_meta_frame_number`, `*_meta_exposure`, `*_meta_gpio_input_data`: Orbbec metadata for checking frame sequence, exposure changes, and trigger/GPIO state.
- `*_meta_type_0` ... `*_meta_type_33`: all raw `OBFrameMetadataType` values exposed by `hasMetadata()` / `getMetadataValue()`.

Missing Orbbec metadata is written as `-1`.  Compare `rgb_raw_ts_us`,
`raw_rgb_meta_sensor_timestamp`, and `raw_rgb_meta_frame_number` first when
checking whether the saved RGB visual content is offset from the timestamp used
for event pairing.

If the program reports `Orbbec camera initialization failed: No device found`, first verify that Windows Device Manager sees the camera, then rerun the metadata registration script above. This error usually means the SDK cannot enumerate a connected device, not that the RGBD pipeline logic is broken.
