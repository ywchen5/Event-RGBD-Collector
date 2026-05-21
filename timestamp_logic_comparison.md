# Timestamp Logic Comparison

This note compares the current `Event-RGBD-Collector` workspace with
`D:\ywchen\Event-RGBD-Collector-upstream-tmp`.

## Conclusion

The current `images.txt` timestamp recording is not inherited from the upstream
initial version. The upstream version did not write `frame/images.txt` at all.

The previously observed behavior was introduced by local commit:

```text
f6e3f44 Add ts information for rgbd
```

After that change, each RGB/depth line in `images.txt` was written with the
matched event slice's `startTs/endTs`, not the Orbbec RGB/depth frame's own
`colorTimestampUs/depthTimestampUs`.

This workspace has now been corrected so `images.txt` records the actual
Orbbec timestamp for each image:

- RGB line: `colorTimestampUs colorTimestampUs`
- depth line: `depthTimestampUs depthTimestampUs`

Therefore, when a CSV shows:

```text
event_minus_rgb = 0
```

it only proves that the RGB/depth metadata line was labeled with the same event
slice timestamp. It does not prove that the RGB image content and the event h5
content were physically captured at the same time.

## Upstream Behavior

In `Event-RGBD-Collector-upstream-tmp`:

- Orbbec frames already have real SDK timestamps:
  - `colorFrame->timeStampUs()`
  - `depthFrame->timeStampUs()`
- Prophesee events already have real event timestamps:
  - `EventCD::t`
  - external trigger `EventExtTrigger::t`
- h5 event files already write event slice attributes:
  - `start_ts`
  - `end_ts`
- There is no `images.txt` output.

So upstream stores event h5 timing, but does not store RGB/depth timing metadata
beside the image files.

## Current Timestamp Sources

### Orbbec RGB/depth timestamps

In `src/orbbec_processor.cpp`, Orbbec timestamps come from the SDK:

```cpp
uint64_t oColorTs = colorFrame->timeStampUs();
uint64_t oDepthTs = depthFrame->timeStampUs();
```

They are stored in:

```cpp
frame.colorTimestampUs = oColorTs;
frame.depthTimestampUs = oDepthTs;
```

These are the real Orbbec device timestamps.

### Event timestamps

In `src/prophesee_processor.cpp`, event timestamps come from Metavision:

```cpp
static_cast<int64_t>(it->t)
```

The event slice boundaries come from external trigger rising edges:

```cpp
slice.startTs = lastTriggerTs_;
slice.endTs   = currentTrigTs;
```

These are on the Prophesee/event-camera timestamp timeline.

### Event h5 timestamps

In `src/data_recorder.cpp`, h5 attributes are written from the event slice:

```cpp
int64_t startTs = task.events.startTs;
int64_t endTs   = task.events.endTs;
```

This is consistent with the h5 content.

## Current `images.txt` Behavior

The problematic previous code copied event slice timestamps into the image
writer task:

```cpp
imgTask.evStartTs = task.events.startTs;
imgTask.evEndTs   = task.events.endTs;
```

Then `images.txt` was written using those values:

```cpp
uint64_t evStart = task.evStartTs;
uint64_t evEnd   = task.evEndTs;
```

So the output looked like:

```text
000498_rgb.jpg   <event_start_ts> <event_end_ts>
000498_depth.png <event_start_ts> <event_end_ts>
```

This was the key mismatch: the filename was RGB/depth, but the timestamps
written beside it were event slice timestamps.

The corrected logic writes the Orbbec timestamp directly:

```cpp
uint64_t ts = task.orbbec.colorTimestampUs;
ofs << rgb_filename << " " << ts << " " << ts << "\n";

uint64_t ts = task.orbbec.depthTimestampUs;
ofs << depth_filename << " " << ts << " " << ts << "\n";
```

## Sync Logic Comparison

The upstream and current code both use the same core timestamp-mapping strategy
for pairing:

1. Bootstrap keeps the newest Orbbec frame and newest event slice.
2. It estimates:

```cpp
deltaOrbToEvs_ = evsTs - orbTs;
```

3. For each Orbbec frame, it maps Orbbec timestamp into the event timeline:

```cpp
mappedOrbTs = orbTs + deltaOrbToEvs_;
```

4. It selects the event slice with closest `startTs`.

So the current serious visual mismatch is not caused by a recent change from
sequence pairing to timestamp pairing. The nearest-timestamp pairing strategy
already existed in upstream.

## Current Workspace Changes vs Upstream

Main differences found:

1. `images.txt` writing was added locally.
   - Upstream does not write it.
   - Current `images.txt` records event timestamps for RGB/depth filenames.

2. A pair queue was added locally.
   - Upstream used only `getLatestPair()`, which exposes the latest pair and can
     skip intermediate pairs if the main loop is slower.
   - Current workspace adds `popPair()` and `pairQueue_`, then drains all queued
     pairs in `main.cpp`.
   - This change is intended to reduce dropped synced pairs and does not change
     the actual timestamp values written by `DataRecorder`.

3. Orbbec initialization was wrapped in error handling.
   - This affects robustness when the device fails to open.
   - It does not affect timestamp semantics.

4. h5 event timestamp logic is unchanged in meaning.
   - It still records event slice `startTs/endTs`.

## What This Means for the LED Grid Mismatch

The LED-grid observation is:

- Around frame 498, RGB/projection already shows VSCode.
- The event image still shows LED grid circles until the end.

Given the current metadata logic, `images.txt` cannot validate this away,
because RGB entries are labeled with event timestamps.

The mismatch must be checked using real per-pair metadata:

- pair index
- Orbbec `colorTimestampUs`
- Orbbec `depthTimestampUs`
- event `startTs`
- event `endTs`
- `clockDiffUs`
- number of events
- file index actually written

Without saving these fields, the current dataset can only prove that files with
the same numeric prefix were written together by the recorder. It cannot prove
that the visual content in the RGB image and h5 event slice are physically
synchronous.

## Recommended Fix

Keep `images.txt` for image-native timestamps. If event/RGB pairing diagnostics
are needed, add a separate `pairs.csv` to record both timelines explicitly. For
example:

```text
filename event_start_us event_end_us orbbec_color_ts_us orbbec_depth_ts_us clock_diff_us seq
000498_rgb.jpg 16759852 16793183 45123456789 45123490120 -12 498
000498_depth.png 16759852 16793183 45123456789 45123490120 -12 498
```

Also save a separate `pairs.csv` once per synced pair:

```text
idx,seq,event_file,rgb_file,depth_file,event_start_us,event_end_us,color_ts_us,depth_ts_us,clock_diff_us,event_count
```

This will make it possible to distinguish:

- metadata labeling problem
- sync pairing problem
- recorder queue/drop problem
- projection/visualization problem
