# 🚀 快速開始指南

## 5 分鐘快速上手

### 1. 安裝依賴
```bash
# Linux/macOS: 自動安裝 (推薦)
./install.sh

# Windows: 自動安裝 (推薦)
install.bat

# 或手動安裝
pip install -r requirements.txt
# 然後安裝 FFmpeg (參考 README.md)
```

### 2. 準備音訊檔案
```bash
# Linux/macOS: 將您的音訊檔案放在專案根目錄
cp /path/to/your/audio.aac source.aac

# Windows: 複製音訊檔案到專案根目錄
copy "C:\path\to\your\audio.aac" source.aac
```

### 3. 執行轉錄
```bash
# Linux/macOS: 一鍵轉錄
python3 transcribe.py

# Windows: 一鍵轉錄
python transcribe.py
```

### 4. 查看結果
```bash
# Linux/macOS: 結果會自動保存在 轉錄結果/ 目錄
ls 轉錄結果/
cat 轉錄結果/result-source-*.txt

# Windows: 結果會自動保存在 轉錄結果\ 目錄
dir 轉錄結果\
type 轉錄結果\result-source-*.txt
```

## 支援的音訊格式

- ✅ `.aac` - AAC 音訊編碼
- ✅ `.mp3` - MP3 音訊編碼  
- ✅ `.wav` - WAV 無損音訊
- ✅ `.m4a` - M4A 音訊編碼
- ✅ `.flac` - FLAC 無損音訊
- ✅ `.ogg` - OGG 音訊編碼

## 硬體需求

### 最低需求
- Python 3.8+
- 4GB RAM
- 2GB 可用儲存空間

### 推薦配置
- Python 3.9+
- 8GB+ RAM
- Windows: NVIDIA GPU (RTX 系列) 或 Intel/AMD CPU
- macOS: Apple Silicon (M1/M2/M3/M4) 或 Intel CPU
- Linux: NVIDIA GPU 或 Intel/AMD CPU
- 5GB+ 可用儲存空間

## 常見問題

### Q: 轉錄結果只有驚嘆號？
A: 可能是音訊品質問題，程式會自動嘗試備用模型

### Q: 記憶體不足？
A: 程式會自動調整參數，建議關閉其他應用程式

### Q: 處理速度慢？
A: 確保使用 Apple Silicon 或 NVIDIA GPU 加速

### Q: FFmpeg 錯誤？
A: 執行 `brew install ffmpeg` (macOS) 或 `sudo apt install ffmpeg` (Linux)

## 進階功能

### 測試特定段落
```bash
python3 transcribe.py test
```

### 自定義設定
編輯 `transcribe.py` 中的參數

### 查看詳細文檔
- [完整 README](README.md)
- [使用範例](examples/usage_examples.md)
- [專案需求](專案需求.md)

## 需要幫助？

1. 查看 [Issues](https://github.com/your-username/smart-audio-transcriber/issues)
2. 閱讀 [使用範例](examples/usage_examples.md)
3. 創建新的 Issue

---

**⭐ 如果這個工具對您有幫助，請給我們一個 Star！**
