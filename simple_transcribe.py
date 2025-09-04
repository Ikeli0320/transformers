#!/usr/bin/env python3
"""
簡化版轉錄程式 - 專注於解決轉錄結果空白問題
"""

import os
import torch
from transformers import pipeline
import subprocess
from datetime import datetime

def main():
    print("🚀 啟動簡化版轉錄程式")
    print("=" * 50)
    
    # 檢查硬體
    print(f"🔧 CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    
    # 載入模型
    print("🤖 載入 Breeze-ASR-25 模型...")
    model = pipeline(
        task="automatic-speech-recognition",
        model="MediaTek-Research/Breeze-ASR-25",
        device=0 if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        return_timestamps=True
    )
    print("✅ 模型載入完成")
    
    # 創建輸出檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"轉錄結果/result-source-{timestamp}.txt"
    
    # 寫入標題
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("智能語音轉錄結果 (Breeze-ASR-25)\n")
        f.write("=" * 60 + "\n")
        f.write(f"檔案: source.aac\n")
        f.write(f"模型: MediaTek-Research/Breeze-ASR-25\n")
        f.write(f"硬體: {'NVIDIA GPU' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"轉錄時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("轉錄結果:\n")
        f.write("=" * 60 + "\n")
    
    print(f"📁 輸出檔案: {output_file}")
    
    # 分段處理
    segment_duration = 300  # 5分鐘一段
    total_duration = 4153  # 總長度（秒）
    num_segments = (total_duration // segment_duration) + 1
    
    print(f"🔢 將分為 {num_segments} 個段落，每段 {segment_duration} 秒")
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        
        print(f"📊 處理段落 {i+1}/{num_segments}: {start_time}s - {end_time}s")
        
        try:
            # 提取音訊段落
            segment_file = f"temp/segment_{i}.wav"
            cmd = [
                'ffmpeg', '-i', 'source.aac',
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-ar', '16000',
                '-ac', '1',
                '-y', segment_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  段落 {i+1} 提取失敗")
                continue
            
            # 轉錄
            print(f"🎯 轉錄段落 {i+1}...")
            transcription_result = model(segment_file, return_timestamps=True)
            
            print(f"🔍 轉錄結果: {transcription_result}")
            
            # 保存結果
            with open(output_file, "a", encoding="utf-8") as f:
                if 'text' in transcription_result and transcription_result['text'].strip():
                    text = transcription_result['text'].strip()
                    f.write(f"[段落 {i+1}] {text}\n")
                    print(f"✅ 已保存: {text[:50]}...")
                else:
                    f.write(f"[段落 {i+1}] (無語音內容)\n")
                    print(f"⚠️  段落 {i+1} 無語音內容")
            
            # 清理臨時檔案
            if os.path.exists(segment_file):
                os.remove(segment_file)
                
        except Exception as e:
            print(f"❌ 段落 {i+1} 處理失敗: {e}")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"[段落 {i+1}] 處理失敗: {str(e)}\n")
            continue
    
    print("=" * 50)
    print("🎉 轉錄完成！")
    print(f"📁 結果保存在: {output_file}")

if __name__ == "__main__":
    main()
