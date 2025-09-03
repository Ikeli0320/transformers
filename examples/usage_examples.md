# 使用範例 (Usage Examples)

## 基本使用

### 1. 準備音訊檔案
```bash
# Linux/macOS: 將您的音訊檔案放在專案根目錄
cp /path/to/your/audio.aac source.aac

# Windows: 複製音訊檔案到專案根目錄
copy "C:\path\to\your\audio.aac" source.aac
```

### 2. 執行轉錄
```bash
# Linux/macOS: 自動轉錄
python3 transcribe.py

# Windows: 自動轉錄
python transcribe.py
```

### 3. 查看結果
```bash
# Linux/macOS: 結果會保存在 轉錄結果/ 目錄
ls 轉錄結果/
cat 轉錄結果/result-source-*.txt

# Windows: 結果會保存在 轉錄結果\ 目錄
dir 轉錄結果\
type 轉錄結果\result-source-*.txt
```

## 進階使用

### 測試特定段落
```bash
# 測試 12-22 秒的段落
python3 transcribe.py test
```

### 自定義測試時間
修改 `transcribe.py` 中的測試函數：
```python
def test_audio_segment(audio_path, start_time=10, duration=10):
    # 修改 start_time 和 duration 參數
    test_audio_segment("source.aac", start_time=30, duration=20)
```

## 支援的音訊格式

### 輸入格式
- `.aac` - AAC 音訊編碼
- `.mp3` - MP3 音訊編碼
- `.wav` - WAV 無損音訊
- `.m4a` - M4A 音訊編碼
- `.flac` - FLAC 無損音訊
- `.ogg` - OGG 音訊編碼

### 輸出格式
- `.txt` - 純文字轉錄結果
- 包含時間戳的分段轉錄

## 硬體優化範例

### Apple Silicon (M1/M2/M3/M4)
```bash
# 自動偵測並使用 MPS 加速
python3 transcribe.py
# 輸出: ✅ 使用 Apple Silicon MPS 加速
```

### NVIDIA GPU
```bash
# 自動偵測並使用 CUDA 加速
python3 transcribe.py
# 輸出: ✅ 使用 NVIDIA CUDA 加速
```

### CPU 模式
```bash
# 自動回退到 CPU 處理
python3 transcribe.py
# 輸出: ✅ 使用 CPU 處理
```

## 記憶體優化範例

### 高記憶體系統 (16GB+)
```
智能分段大小: 300秒, 重疊: 10秒
批次大小: 2
精度: torch.float16
```

### 中等記憶體系統 (8-16GB)
```
智能分段大小: 180秒, 重疊: 5秒
批次大小: 1
精度: torch.float16
```

### 低記憶體系統 (<8GB)
```
智能分段大小: 60秒, 重疊: 3秒
批次大小: 1
精度: torch.float32
```

## 音訊預處理範例

### 空白段落移除
```
🔍 偵測空白段落 (閾值: -30dB, 最小長度: 1.0秒)...
📊 偵測到 5 個空白段落
✂️  移除空白段落並合併有效音訊...
📊 檔案大小減少: 23.5%
```

### 音量增強
```
音訊格式: aac
採樣率: 48000 Hz
聲道數: 1
時長: 3600.0 秒
音量: -32.5 dB
動態音量增強: 26.5 dB
```

## 模型切換範例

### 主要模型 (Breeze-ASR-25)
```
🤖 智能載入語音轉錄模型...
🎯 主要模型: Breeze-ASR-25 (台灣中文優化)
✅ Breeze-ASR-25 模型載入完成！
```

### 備用模型 (Whisper)
```
⚠️  主要模型無法識別內容，嘗試備用模型...
🔄 切換到 Whisper 模型...
✅ 使用 Whisper 模型重新轉錄
```

## 分段處理範例

### 實時分段轉錄
```
📊 開始分段轉錄 (180秒/段)...
🔄 處理段落 1/20 (0.0s - 180.0s)...
✅ 段落 1 轉錄完成
🔄 處理段落 2/20 (180.0s - 360.0s)...
✅ 段落 2 轉錄完成
```

### 續轉功能
```
🔍 檢查現有轉錄結果...
📁 找到匹配的結果檔案: result-source-20250903_224449.txt
✅ 檔案大小和時長匹配，續接轉錄
🔄 從段落 15 開始續轉...
```

## 故障排除範例

### 記憶體不足
```
⚠️  記憶體使用率過高，建議關閉其他應用程式
📊 載入後記憶體使用: 92.3%
💡 自動調整參數以適應可用記憶體
```

### FFmpeg 錯誤
```
❌ WAV 轉換失敗: ffmpeg: command not found
💡 請安裝 FFmpeg: brew install ffmpeg
```

### 模型載入失敗
```
⚠️  Breeze-ASR-25 載入失敗: Connection timeout
🔄 切換到備用 Whisper 模型...
✅ Whisper 備用模型載入完成！
```

## 效能監控範例

### 進度顯示
```
🔄 轉錄進度: 45% (段落 9/20)
⏱️  已用時間: 12分30秒
📊 預估剩餘: 15分20秒
💾 記憶體使用: 73.2%
```

### 完成統計
```
🎯 轉錄完成！
📊 總處理時間: 27分50秒
📁 結果檔案: 轉錄結果/result-source-20250903_230000.txt
💾 最終記憶體使用: 68.5%
```

## faster-whisper 整合範例

### 自動 faster-whisper 支援
```python
# 程式會自動偵測並使用 faster-whisper
# 如果可用，會優先使用 faster-whisper 作為備用模型
# 效能提升 4-5 倍

# 手動啟用 faster-whisper
pip install faster-whisper
python3 transcribe.py
```

### Colab 版本使用
```python
# 在 Google Colab 中使用
# 參考 examples/colab_example.py

# 上傳到 Colab 並執行
# 自動偵測 GPU/CPU 並優化參數
# 支援 VAD 和智能空白段落移除
```

## 自定義設定範例

### 修改分段大小
```python
# 在 transcribe.py 中修改
def _optimize_parameters(self):
    # 自定義分段大小
    if available_memory_gb >= 16:
        segment_duration = 600  # 10分鐘分段
    elif available_memory_gb >= 8:
        segment_duration = 300  # 5分鐘分段
    else:
        segment_duration = 180  # 3分鐘分段
```

### 修改音量增強
```python
# 在 preprocess_audio 方法中修改
volume_boost = min(volume_boost, 40)  # 最大增強 40dB
```

### 修改空白偵測閾值
```python
# 在 _detect_silence_segments 方法中修改
silence_segments = self._detect_silence_segments(
    temp_wav, 
    silence_threshold=-25,  # 更敏感的空白偵測
    min_silence_duration=0.5  # 更短的空白段落
)
```
