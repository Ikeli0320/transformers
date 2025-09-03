@echo off
REM 智能語音轉錄工具 Windows 安裝腳本
REM Smart Audio Transcriber Windows Installation Script

echo 🎯 智能語音轉錄工具 Windows 安裝腳本
echo ==================================

REM 檢查 Python 版本
echo 🔍 檢查 Python 版本...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 未安裝，請先安裝 Python 3.8+
    echo 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo ✅ Python 已安裝

REM 檢查 pip
echo 🔍 檢查 pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip 未安裝，請先安裝 pip
    pause
    exit /b 1
)

echo ✅ pip 已安裝

REM 檢查 FFmpeg
echo 🔍 檢查 FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  FFmpeg 未安裝
    echo 📦 請手動安裝 FFmpeg:
    echo    1. 前往 https://ffmpeg.org/download.html
    echo    2. 下載 Windows 版本
    echo    3. 解壓縮並添加到系統 PATH
    echo    4. 或使用 Chocolatey: choco install ffmpeg
    echo    5. 或使用 Scoop: scoop install ffmpeg
    echo.
    echo 安裝完成後請重新執行此腳本
    pause
    exit /b 1
) else (
    echo ✅ FFmpeg 已安裝
)

REM 安裝 Python 依賴
echo 📦 安裝 Python 依賴...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Python 依賴安裝失敗
    pause
    exit /b 1
)

echo ✅ Python 依賴安裝完成

REM 檢查 faster-whisper 安裝
echo 🔍 檢查 faster-whisper 支援...
python -c "try: from faster_whisper import WhisperModel; print('✅ faster-whisper 已安裝，將提供 4-5x 效能提升') except ImportError: print('💡 faster-whisper 未安裝，將使用標準 transformers'); print('   如需更高效能，可執行: pip install faster-whisper')"

REM 創建必要的目錄
echo 📁 創建必要的目錄...
if not exist "轉錄結果" mkdir "轉錄結果"
if not exist "temp" mkdir "temp"

REM 測試安裝
echo 🧪 測試安裝...
python -c "import torch; import transformers; print('✅ PyTorch 版本:', torch.__version__); print('✅ Transformers 版本:', transformers.__version__); if torch.cuda.is_available(): print('✅ NVIDIA CUDA 支援可用'); else: print('💡 將使用 CPU 處理')"

if %errorlevel% equ 0 (
    echo.
    echo 🎉 安裝完成！
    echo ==================================
    echo 📖 使用方法:
    echo 1. 將音訊檔案放在專案根目錄並命名為 source.aac
    echo 2. 執行: python transcribe.py
    echo 3. 查看結果: dir 轉錄結果\
    echo.
    echo 📚 更多資訊請查看 README.md
    echo 🐛 問題回報請查看 examples\usage_examples.md
) else (
    echo ❌ 安裝測試失敗
    pause
    exit /b 1
)

pause
