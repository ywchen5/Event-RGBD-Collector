# 采集同步问题现象与原因分析

## 问题现象

当前采集得到的 RGB、Depth、Event 文件数量是一一对应的，看起来每组数据都保存成功。例如某组数据中：

```text
RGB   2454
Depth 2454
Event 2454
```

但检查 `frame/images.txt` 中记录的时间戳后发现，帧间隔并不稳定在 30fps 对应的 `33331/33332us`，而是大量出现：

```text
66662us
99993us
133324us
```

这说明虽然文件数量是成组的，但保存下来的序列并不是连续 30fps，而是存在跳帧或抽帧现象。

该问题会影响 blinking pattern 测试和动态场景对齐判断。因为中间帧被跳过后，闪烁相位和运动状态会出现不连续，从而表现为 RGBD 与 event 在部分帧中看起来不同步或不稳定。

## 原因分析

当前采集链路中，Orbbec 和 Prophesee 各自内部是队列式缓存，`SyncProcessor` 也会持续配对 RGBD 和 event slice。

但 `SyncProcessor` 对主线程输出时，只保留了一个“最新 pair”：

```cpp
pairFront_ = pair;
newPairReady_ = true;
```

主线程通过 `getLatestPair()` 取数据。如果主线程因为显示、图像解码、resize、`imshow` 或系统调度没有及时读取，那么 `SyncProcessor` 新生成的 pair 会覆盖旧 pair。

因此，中间已经同步好的数据可能没有进入 `DataRecorder`，最终保存序列就会出现 `66ms`、`99ms`、`133ms` 等间隔。

也就是说，问题不是 RGB/depth/event 文件数量不匹配，而是同步后的 pair 在进入保存模块前被 latest-buffer 覆盖，导致保存序列不连续。

## 结论

当前问题的关键不是重投影算法，也不是单纯标定误差，而是采集保存链路中存在丢中间 pair 的风险。

需要把 `SyncProcessor` 到 `DataRecorder` 的传递方式从“只取最新帧”改成 FIFO 队列，确保每一个同步 pair 都能按顺序进入保存模块。

推荐方向：

```text
SyncProcessor 生成 pair
        ↓
pairQueue_ FIFO 缓存
        ↓
DataRecorder 按顺序 pop/consume
```

显示模块可以继续使用最新帧，但保存模块必须消费完整 FIFO 队列。

## 修复方案

### 方案目标

修复的核心目标是：保存模块必须拿到每一个已经同步成功的 pair，而不是只拿到主线程来得及读取的“最新 pair”。

也就是说：

- 采集与同步线程保持 30fps 产生 pair。
- 保存链路按 FIFO 顺序消费 pair。
- 显示链路可以丢帧，只显示最新画面。
- 保存链路不能因为显示慢而丢同步数据。

### 方案一：在 SyncProcessor 中增加 FIFO pair 队列

这是推荐方案。

当前逻辑：

```text
SyncProcessor
    pairFront_ = latest pair
        ↓
main thread getLatestPair()
        ↓
DataRecorder enqueue()
```

问题是 `pairFront_` 只有一个槽位，会被新 pair 覆盖。

修复后：

```text
SyncProcessor
    pairQueue_.push_back(pair)
        ↓
main thread popPair()
        ↓
DataRecorder enqueue()
```

同时保留 `pairFront_` 或 `latestPair_` 供显示使用。

建议接口：

```cpp
bool SyncProcessor::popPair(SyncedPair &out);
bool SyncProcessor::getLatestPair(SyncedPair &out);
size_t SyncProcessor::pairQueueSize() const;
```

其中：

- `popPair()`：FIFO 取出最旧的同步 pair，用于保存。
- `getLatestPair()`：只用于显示或监控，可以丢帧。
- `pairQueueSize()`：用于日志监控，判断保存端是否跟不上。

### 方案二：让 DataRecorder 通过 callback 直接接收 pair

当前 `SyncProcessor` 已经有 callback 机制：

```cpp
setCallback(PairCallback cb);
```

可以在生成 pair 后直接调用：

```cpp
callback_(pair);
```

然后在 callback 里执行：

```cpp
recorder->enqueue(pair);
```

优点是改动较小，绕过主线程显示逻辑。

但缺点是：`recorder->enqueue(pair)` 会复制 RGBD 和 event 数据，如果 callback 在 Sync 线程中执行，可能阻塞同步线程。因此该方案需要确保 `enqueue()` 足够轻，或者 callback 只做轻量转发。

因此更推荐方案一，即 `SyncProcessor` 内部维护 FIFO 队列。

### 推荐实现细节

1. 在 `SyncProcessor` 中新增成员：

```cpp
std::deque<SyncedPair> pairQueue_;
mutable std::mutex pairQueueMutex_;
static constexpr size_t MAX_PAIR_QUEUE = 300;
std::atomic<uint64_t> pairQueueDrops_{0};
```

2. 每次生成同步 pair 时：

```cpp
{
    std::lock_guard<std::mutex> lock(pairQueueMutex_);
    if (pairQueue_.size() >= MAX_PAIR_QUEUE) {
        pairQueue_.pop_front();
        pairQueueDrops_++;
    }
    pairQueue_.push_back(pair);
}
```

3. 新增 FIFO 读取接口：

```cpp
bool SyncProcessor::popPair(SyncedPair &out) {
    std::lock_guard<std::mutex> lock(pairQueueMutex_);
    if (pairQueue_.empty()) return false;
    out = std::move(pairQueue_.front());
    pairQueue_.pop_front();
    return true;
}
```

4. 修改 `main.cpp` 中保存逻辑：

当前：

```cpp
if (sync->getLatestPair(pair) && pair.valid) {
    if (recorder) {
        recorder->enqueue(pair);
    }
}
```

建议改为：

```cpp
bool recordedAny = false;
while (sync->popPair(pair)) {
    if (pair.valid && recorder) {
        recorder->enqueue(pair);
        recordedAny = true;
    }
}
```

5. 显示逻辑单独使用 latest pair。

保存逻辑应该尽量独立于显示逻辑。即使显示窗口卡顿，也不应该影响保存。

可以采用：

```text
while (sync->popPair(pair)) {
    recorder->enqueue(pair);
    latestForDisplay = pair;
}

if (displayEnabled && latestForDisplay.valid) {
    update display;
}
```

这样保存会消费所有 pair，显示只使用最新一帧。

### images.txt 顺序问题

当前 `DataRecorder` 使用多个 image worker 并行写图，因此 `images.txt` append 顺序可能和文件编号顺序不同。

建议两种处理方式：

#### 方式 A：分析时按文件名排序

这是最小改动。后处理检查 `images.txt` 时，按 `000000_rgb.jpg`、`000001_rgb.jpg` 排序后再计算时间差。

#### 方式 B：改为最终统一写 metadata

更稳妥的方式是在 `DataRecorder` 中维护一个 metadata 队列或 vector，记录每个 `idx` 的：

```text
idx
rgb filename
depth filename
event filename
event start_ts
event end_ts
orbbec color_ts
orbbec depth_ts
```

停止采集时统一按 `idx` 排序写出 `metadata.csv` 或 `images.txt`。

这可以避免多线程 append 导致的行顺序混乱。

### 验证方法

修复后需要做三组验证：

1. 不开显示采集：

```powershell
.\scanner_cpp_optim.exe --store --no-display --output D:\ywchen\data_out\test_fifo
```

检查 `images.txt`，按文件名排序后，帧间隔应接近：

```text
33331
33332
33331
33332
```

2. 开显示采集：

```powershell
.\scanner_cpp_optim.exe --store --output D:\ywchen\data_out\test_fifo_display
```

如果 FIFO 修复有效，即使显示偶尔卡顿，保存结果仍应保持连续 30fps。

3. blinking pattern 测试：

使用闪烁图案检查 RGBD 与 event 的相位是否稳定。如果帧间隔连续，仍出现固定偏移，再考虑外参或曝光时序问题。

### 判断标准

修复成功的标准：

```text
RGB count   == Depth count == Event count
帧间隔主要为 33331/33332us
不再大量出现 66662/99993/133324us
pairQueueDrops == 0
Recorder HDF5/Image drops == 0
```

如果仍然大量出现 `66662us` 以上间隔，则说明问题不只在 latest-buffer，还需要继续检查外部 trigger、Orbbec SECONDARY 出帧和 Prophesee trigger slicing。
