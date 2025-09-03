# Colab 版本 - 智能語音轉錄工具
# 基於 faster-whisper 的高效能版本

"""
Colab 智能語音轉錄工具 (基於 faster-whisper)
────────────────────────────────────────
1. 安裝依賴（僅第一次執行需要）
2. 將音檔拖到 Colab 左側 Files 或用 upload 視窗上傳
3. 自動轉成 16 kHz / mono WAV ➜ faster-Whisper ➜ .txt
Author: Smart Audio Transcriber Team
License: MIT
"""

# ── 1. 安裝系統與 Python 套件 ────────────────────
print("Installing dependencies...")
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", package])

try:
    # Check if ffmpeg is installed by trying to run it
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True)
    print("ffmpeg is already installed.")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Installing ffmpeg...")
    subprocess.run(["apt-get", "-y", "update"], check=True)
    subprocess.run(["apt-get", "-y", "install", "-y", "ffmpeg"], check=True)
    print("ffmpeg installed.")

install("pydub")
install("faster-whisper")
install("psutil")
print("Dependencies installed/updated.")

# ── 2. 匯入所需函式庫 ────────────────────────────
import os
import tempfile
import torch
import psutil
from typing import Optional

from google.colab import files
from pydub import AudioSegment
from faster_whisper import WhisperModel

print("Libraries imported.")

# ── 3. 工具函式 ──────────────────────────────────
def detect_hardware():
    """智能硬體偵測"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device = "cuda"
        acceleration = "NVIDIA CUDA"
        compute_type = "float16"
        model_size = "large-v3"
    else:
        device = "cpu"
        acceleration = "CPU"
        compute_type = "int8"
        model_size = "medium"
    
    print(f"🔧 硬體偵測:")
    print(f"   記憶體: {memory_gb:.1f} GB")
    print(f"   加速方式: {acceleration}")
    print(f"   推薦模型: {model_size}")
    print(f"   計算類型: {compute_type}")
    
    return {
        "device": device,
        "acceleration": acceleration,
        "compute_type": compute_type,
        "model_size": model_size,
        "memory_gb": memory_gb
    }

def convert_to_wav(src_path: str, target_sr: int = 16_000) -> str:
    """
    將任何格式音檔轉成 16 kHz・mono・16-bit PCM WAV。
    回傳暫存檔路徑，處理過後自動刪除原檔不會影響。
    """
    print(f"Converting {src_path} to WAV...")
    audio = AudioSegment.from_file(src_path)
    audio = (
        audio.set_frame_rate(target_sr)
             .set_channels(1)
             .set_sample_width(2)      # 16-bit
    )
    # Using a temporary file that pydub can write to directly
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name
    audio.export(wav_path, format="wav")
    print(f"Converted to WAV: {wav_path}")
    return wav_path

def transcribe_optimized(
    model: WhisperModel,
    wav_path: str,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 500
) -> dict:
    """
    使用 faster-Whisper 轉錄音檔。
    """
    print("🔊 Transcribing with faster-whisper...")
    segments, info = model.transcribe(
        wav_path,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=vad_min_silence_duration_ms) if vad_filter else None
    )

    print(f"Detected language '{info.language}' with probability {info.language_probability}")
    print(f"Transcription duration: {info.duration}s (approx.)")

    # 收集所有段落
    all_segments = list(segments)
    full_transcript = "".join(segment.text for segment in all_segments)
    
    # 創建帶時間戳的結果
    result = {
        "text": full_transcript,
        "chunks": []
    }
    
    for segment in all_segments:
        result["chunks"].append({
            "text": segment.text,
            "timestamp": [segment.start, segment.end]
        })

    return result

def save_transcript(result: dict, src_filename: str) -> str:
    """
    以來源檔名為基礎，儲存 *_transcript.txt，並回傳檔名。
    """
    stem = os.path.splitext(os.path.basename(src_filename))[0]
    txt_path = f"{stem}_transcript_smart.txt"
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("智能語音轉錄結果 (Smart Audio Transcriber)\n")
        f.write("=" * 60 + "\n")
        f.write(f"來源檔案: {src_filename}\n")
        f.write(f"轉錄時間: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"模型: faster-whisper\n")
        f.write("=" * 60 + "\n\n")
        
        # 完整轉錄結果
        f.write("完整轉錄結果:\n")
        f.write("-" * 40 + "\n")
        f.write(result["text"])
        f.write("\n\n")
        
        # 分段時間戳
        if result["chunks"]:
            f.write("分段時間戳:\n")
            f.write("-" * 40 + "\n")
            for chunk in result["chunks"]:
                start_time = chunk["timestamp"][0]
                end_time = chunk["timestamp"][1]
                text = chunk["text"]
                f.write(f"[{start_time:.1f}s - {end_time:.1f}s] {text}\n")
    
    print(f"💾 Saved transcript → {txt_path}")
    return txt_path

# ── 4. 主流程 ────────────────────────────────────
def main_smart_transcriber(
    src_path: Optional[str] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 500
) -> None:
    """
    智能語音轉錄主流程
    """
    # 4-0 智能硬體偵測
    hardware_info = detect_hardware()
    
    # 使用偵測結果或用戶指定參數
    global_device = device or hardware_info["device"]
    global_compute_type = compute_type or hardware_info["compute_type"]
    global_model_size = model_size or hardware_info["model_size"]
    
    print(f"\n⏳ Loading faster-Whisper {global_model_size} model...")
    print(f"   Device: {global_device}")
    print(f"   Compute Type: {global_compute_type}")
    
    try:
        model = WhisperModel(global_model_size, device=global_device, compute_type=global_compute_type)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Trying fallback configuration...")
        
        # 嘗試備用配置
        try:
            model = WhisperModel("medium", device="cpu", compute_type="int8")
            print("✅ Fallback model loaded successfully.")
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return

    # 4-1 選取音檔
    if src_path is None:
        print("\nPlease upload an audio file.")
        uploads = files.upload()
        if not uploads:
            print("❌ 沒有檔案！請至少上傳一個音檔。")
            return
        src_path = list(uploads.keys())[0]
        print(f"File '{src_path}' uploaded and saved to Colab environment.")

    if not os.path.exists(src_path):
        print(f"❌ 找不到檔案：{src_path}")
        return

    print(f"\n📥 Source file: {src_path}")

    temp_wav_path = None
    try:
        # 4-2 轉成 WAV
        temp_wav_path = convert_to_wav(src_path)

        # 4-3 faster-whisper 轉錄
        result = transcribe_optimized(
            model,
            temp_wav_path,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms
        )

        # 4-4 儲存・下載
        txt_path = save_transcript(result, src_path)
        print("\n==== 轉錄結果 ====\n")
        print(result["text"])
        print(f"\n📊 總共 {len(result['chunks'])} 個段落")
        files.download(txt_path)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # Clean up the temporary WAV file
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                print(f"Cleaned up temporary WAV file: {temp_wav_path}")
            except OSError as e:
                print(f"Error deleting temporary file {temp_wav_path}: {e}")

# ── 5. 執行 ─────────────────────────────────────
# 智能語音轉錄 - 自動偵測硬體並優化參數
print("🎯 智能語音轉錄工具 (Colab 版本)")
print("=" * 50)

# 執行智能轉錄
main_smart_transcriber(
    src_path=None,  # 使用上傳視窗
    model_size=None,  # 自動選擇
    device=None,  # 自動偵測
    compute_type=None,  # 自動選擇
    beam_size=5,
    vad_filter=True,
    vad_min_silence_duration_ms=500
)
