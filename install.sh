#!/bin/bash

# 智能語音轉錄工具安裝腳本
# Smart Audio Transcriber Installation Script

echo "🎯 智能語音轉錄工具安裝腳本"
echo "=================================="

# 檢查 Python 版本
echo "🔍 檢查 Python 版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 未安裝，請先安裝 Python 3.8+"
    exit 1
fi

# 檢查 pip
echo "🔍 檢查 pip..."
pip3 --version
if [ $? -ne 0 ]; then
    echo "❌ pip 未安裝，請先安裝 pip"
    exit 1
fi

# 檢查 FFmpeg
echo "🔍 檢查 FFmpeg..."
ffmpeg -version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  FFmpeg 未安裝"
    echo "📦 正在安裝 FFmpeg..."
    
    # 檢測作業系統
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ 請先安裝 Homebrew，然後執行: brew install ffmpeg"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt update && sudo apt install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        else
            echo "❌ 請手動安裝 FFmpeg"
            exit 1
        fi
    else
        echo "❌ 不支援的作業系統，請手動安裝 FFmpeg"
        exit 1
    fi
else
    echo "✅ FFmpeg 已安裝"
fi

# 安裝 Python 依賴
echo "📦 安裝 Python 依賴..."
pip3 install -r requirements.txt

# 檢查 faster-whisper 安裝
echo "🔍 檢查 faster-whisper 支援..."
python3 -c "
try:
    from faster_whisper import WhisperModel
    print('✅ faster-whisper 已安裝，將提供 4-5x 效能提升')
except ImportError:
    print('💡 faster-whisper 未安裝，將使用標準 transformers')
    print('   如需更高效能，可執行: pip3 install faster-whisper')
"

if [ $? -eq 0 ]; then
    echo "✅ Python 依賴安裝完成"
else
    echo "❌ Python 依賴安裝失敗"
    exit 1
fi

# 創建必要的目錄
echo "📁 創建必要的目錄..."
mkdir -p 轉錄結果
mkdir -p temp

# 測試安裝
echo "🧪 測試安裝..."
python3 -c "
import torch
import transformers
print('✅ PyTorch 版本:', torch.__version__)
print('✅ Transformers 版本:', transformers.__version__)
if torch.backends.mps.is_available():
    print('✅ Apple Silicon MPS 支援可用')
elif torch.cuda.is_available():
    print('✅ NVIDIA CUDA 支援可用')
else:
    print('💡 將使用 CPU 處理')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 安裝完成！"
    echo "=================================="
    echo "📖 使用方法:"
    echo "1. 將音訊檔案放在專案根目錄並命名為 source.aac"
    echo "2. 執行: python3 transcribe.py"
    echo "3. 查看結果: ls 轉錄結果/"
    echo ""
    echo "📚 更多資訊請查看 README.md"
    echo "🐛 問題回報請查看 examples/usage_examples.md"
else
    echo "❌ 安裝測試失敗"
    exit 1
fi
