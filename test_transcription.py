#!/usr/bin/env python3
"""
簡單的轉錄測試程式
測試硬體、模型和轉錄功能是否正常
"""

import os
import torch
from transformers import pipeline
import subprocess

def test_hardware():
    """測試硬體配置"""
    print("🔧 硬體測試:")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"   PyTorch 版本: {torch.__version__}")
    print()

def test_model_loading():
    """測試模型載入"""
    print("🤖 模型載入測試:")
    try:
        # 測試 Breeze-ASR-25
        print("   載入 Breeze-ASR-25...")
        model = pipeline(
            task="automatic-speech-recognition",
            model="MediaTek-Research/Breeze-ASR-25",
            device=0 if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            return_timestamps=True
        )
        print("   ✅ Breeze-ASR-25 載入成功")
        return model
    except Exception as e:
        print(f"   ❌ Breeze-ASR-25 載入失敗: {e}")
        return None

def create_test_audio():
    """創建測試音訊檔案（15秒）"""
    print("🎵 創建測試音訊檔案:")
    
    # 檢查是否有 source.aac
    if os.path.exists("source.aac"):
        print("   找到 source.aac，提取前 15 秒作為測試")
        cmd = [
            'ffmpeg', '-i', 'source.aac', 
            '-t', '15',  # 只取前 15 秒
            '-ar', '16000',  # 16kHz
            '-ac', '1',      # 單聲道
            '-y', 'test_audio.wav'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ 測試音訊檔案創建成功: test_audio.wav")
            return "test_audio.wav"
        else:
            print(f"   ❌ 測試音訊檔案創建失敗: {result.stderr}")
            return None
    else:
        print("   ❌ 找不到 source.aac 檔案")
        return None

def test_transcription(model, audio_file):
    """測試轉錄功能"""
    print("🎯 轉錄測試:")
    try:
        print(f"   轉錄檔案: {audio_file}")
        result = model(audio_file, return_timestamps=True)
        
        print(f"   轉錄結果: {result}")
        
        if 'text' in result:
            text = result['text'].strip()
            print(f"   轉錄文字: '{text}'")
            print(f"   文字長度: {len(text)} 字元")
            
            if text and text != '!' and len(text) > 2:
                print("   ✅ 轉錄成功！")
                return True
            else:
                print("   ⚠️  轉錄結果異常（只有驚嘆號或太短）")
                return False
        else:
            print("   ❌ 轉錄結果格式錯誤")
            return False
            
    except Exception as e:
        print(f"   ❌ 轉錄失敗: {e}")
        return False

def main():
    print("🚀 開始轉錄功能測試")
    print("=" * 50)
    
    # 1. 測試硬體
    test_hardware()
    
    # 2. 測試模型載入
    model = test_model_loading()
    if not model:
        print("❌ 模型載入失敗，無法繼續測試")
        return
    
    # 3. 創建測試音訊
    audio_file = create_test_audio()
    if not audio_file:
        print("❌ 無法創建測試音訊，無法繼續測試")
        return
    
    # 4. 測試轉錄
    success = test_transcription(model, audio_file)
    
    print("=" * 50)
    if success:
        print("🎉 所有測試通過！轉錄功能正常")
    else:
        print("❌ 轉錄測試失敗，需要檢查問題")
    
    # 清理測試檔案
    if os.path.exists("test_audio.wav"):
        os.remove("test_audio.wav")
        print("🗑️  已清理測試檔案")

if __name__ == "__main__":
    main()
