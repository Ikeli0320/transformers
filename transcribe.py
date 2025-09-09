#!/usr/bin/env python3
"""
智能語音轉錄工具 - 基於 Breeze-ASR-25
自動硬體偵測、動態記憶體管理、智能參數優化
專門針對台灣中文口音優化
"""

import os
import time
import glob
import gc
import threading
import json
import platform
from datetime import datetime
from transformers import pipeline
import psutil
import torch
import subprocess

# 嘗試導入 faster-whisper (可選)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("💡 faster-whisper 未安裝，將使用標準 transformers")

class SmartTranscriber:
    def __init__(self):
        self.supported_formats = ['.aac', '.mp3', '.wav', '.m4a', '.flac']
        self.model = None
        self.output_dir = "轉錄結果"
        self.temp_dir = "temp"
        
        # 進度監控變數
        self.start_time = None
        self.is_processing = False
        self.progress_thread = None
        
        # 智能硬體偵測和參數優化
        self.hardware_info = self._detect_hardware()
        self.optimized_params = self._optimize_parameters()
        
        print(f"🔧 智能硬體偵測完成:")
        print(f"   硬體配置: {self.hardware_info['description']}")
        print(f"   記憶體: {self.hardware_info['memory_gb']:.1f} GB")
        print(f"   可用記憶體: {self.hardware_info['available_memory_gb']:.1f} GB")
        print(f"   加速方式: {self.hardware_info['acceleration']}")
        print(f"   精度: {self.optimized_params['torch_dtype']}")
        print(f"   分段大小: {self.optimized_params['segment_duration']} 秒")
        print(f"   批次大小: {self.optimized_params['batch_size']}")
    
    def _detect_hardware(self):
        """智能硬體偵測"""
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # 偵測處理器架構
        machine = platform.machine().lower()
        system = platform.system().lower()
        
        # 偵測 Apple Silicon
        is_apple_silicon = machine == "arm64" and system == "darwin"
        
        # 偵測 NVIDIA GPU
        has_cuda = torch.cuda.is_available()
        cuda_memory = 0
        if has_cuda:
            cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # 偵測 MPS 支援
        has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        # 選擇最佳加速方式
        # 暫時禁用 MPS，因為可能導致轉錄問題
        if is_apple_silicon and has_mps:
            acceleration = "Apple Silicon (CPU 模式)"
            device = "cpu"
            print("💡 使用 CPU 模式以避免 MPS 相容性問題")
        elif has_cuda:
            acceleration = f"NVIDIA CUDA ({cuda_memory:.1f}GB)"
            device = "cuda"
        else:
            acceleration = "CPU"
            device = "cpu"
        
        # 生成硬體描述
        if is_apple_silicon:
            description = f"Apple Silicon ({machine})"
        elif has_cuda:
            description = f"x86_64 + NVIDIA GPU"
        else:
            description = f"x86_64 CPU"
        
        return {
            'memory_gb': memory_gb,
            'available_memory_gb': available_memory_gb,
            'memory_percent': memory.percent,
            'is_apple_silicon': is_apple_silicon,
            'has_cuda': has_cuda,
            'has_mps': has_mps,
            'cuda_memory': cuda_memory,
            'acceleration': acceleration,
            'device': device,
            'description': description
        }
    
    def _optimize_parameters(self):
        """依據硬體效能自動優化參數"""
        memory_gb = self.hardware_info['available_memory_gb']
        is_apple_silicon = self.hardware_info['is_apple_silicon']
        has_cuda = self.hardware_info['has_cuda']
        
        # 動態調整分段大小 (30-120 秒，更小的分段減少雜訊)
        if memory_gb >= 16:
            segment_duration = 120  # 2 分鐘
        elif memory_gb >= 8:
            segment_duration = 90   # 1.5 分鐘
        else:
            segment_duration = 60   # 1 分鐘
        
        # 動態調整重疊大小 (5-15 秒，增加重疊減少斷句問題)
        stride_duration = min(15, max(5, segment_duration // 20))
        
        # 動態調整批次大小
        if memory_gb >= 16:
            batch_size = 2
        else:
            batch_size = 1
        
        # 選擇精度
        if is_apple_silicon and self.hardware_info['has_mps']:
            torch_dtype = torch.float16
        elif has_cuda and memory_gb >= 8:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # 動態調整音量增強 (5-15dB，更溫和的增強)
        volume_boost = min(15, max(5, 5 + (memory_gb - 4) * 1))
        
        return {
            'segment_duration': segment_duration,
            'stride_duration': stride_duration,
            'batch_size': batch_size,
            'torch_dtype': torch_dtype,
            'volume_boost': volume_boost,
            'chunk_length_s': 30,  # Breeze-ASR-25 官方建議
            'stride_length_s': 5   # Breeze-ASR-25 官方建議
        }
    
    def progress_monitor(self):
        """進度監控線程"""
        last_report = 0
        while self.is_processing:
            if self.start_time:
                elapsed = time.time() - self.start_time
                elapsed_min = elapsed / 60
                
                # 每 30 秒顯示一次進度
                if elapsed - last_report >= 30:
                    current_memory = psutil.virtual_memory().percent
                    
                    # 使用實際處理的段落數計算進度
                    if hasattr(self, 'current_segment') and hasattr(self, 'total_segments'):
                        progress_percent = (self.current_segment / self.total_segments) * 100
                        print(f"📊 進度: {progress_percent:.1f}% | 已處理: {elapsed_min:.1f}分鐘 | 記憶體: {current_memory:.1f}% | 段落: {self.current_segment}/{self.total_segments} | 狀態: 處理中...")
                    else:
                        # 備用：基於預估時間計算進度
                        if hasattr(self, 'estimated_duration_minutes'):
                            progress_percent = min(95, (elapsed_min / self.estimated_duration_minutes) * 100)
                            print(f"📊 進度: {progress_percent:.1f}% | 已處理: {elapsed_min:.1f}分鐘 | 記憶體: {current_memory:.1f}% | 狀態: 處理中...")
                        else:
                            print(f"⏱️  已處理時間: {elapsed_min:.1f} 分鐘 | 記憶體使用: {current_memory:.1f}% | 狀態: 處理中...")
                    
                    last_report = elapsed
            
            time.sleep(1)
    
    def start_progress_monitor(self):
        """開始進度監控"""
        self.is_processing = True
        self.start_time = time.time()
        self.progress_thread = threading.Thread(target=self.progress_monitor, daemon=True)
        self.progress_thread.start()
        print("📊 進度監控線程已啟動")
    
    def stop_progress_monitor(self):
        """停止進度監控"""
        self.is_processing = False
        if self.progress_thread:
            self.progress_thread.join(timeout=1)
        
    def _analyze_audio_quality(self, audio_path):
        """智能音訊品質分析"""
        try:
            # 分析音量
            result = subprocess.run([
                'ffmpeg', '-i', audio_path, '-af', 'volumedetect', '-f', 'null', '-'
            ], capture_output=True, text=True)
            
            volume_info = {}
            for line in result.stderr.split('\n'):
                if 'mean_volume:' in line:
                    volume_info['mean_volume'] = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                elif 'max_volume:' in line:
                    volume_info['max_volume'] = float(line.split('max_volume:')[1].split('dB')[0].strip())
            
            # 分析音訊資訊
            probe_result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', audio_path
            ], capture_output=True, text=True, check=True)
            
            info = json.loads(probe_result.stdout)
            audio_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("找不到音訊流")
            
            return {
                'codec': audio_stream['codec_name'],
                'sample_rate': int(audio_stream['sample_rate']),
                'channels': int(audio_stream['channels']),
                'duration': float(info['format']['duration']),
                'bitrate': int(info['format'].get('bit_rate', 0)),
                'volume': volume_info.get('mean_volume', -20),
                'max_volume': volume_info.get('max_volume', -10)
            }
        except Exception as e:
            print(f"⚠️  音訊分析失敗: {e}")
            return None
    
    def _detect_silence_segments(self, audio_path, silence_threshold=-30, min_silence_duration=1.0):
        """偵測空白段落（溫和設定）"""
        try:
            print(f"🔍 偵測空白段落 (閾值: {silence_threshold}dB, 最小長度: {min_silence_duration}秒)...")
            
            result = subprocess.run([
                'ffmpeg', '-i', audio_path, 
                '-af', f'silencedetect=noise={silence_threshold}dB:duration={min_silence_duration}',
                '-f', 'null', '-'
            ], capture_output=True, text=True)
            
            silence_segments = []
            current_start = None
            for line in result.stderr.split('\n'):
                if 'silence_start:' in line:
                    try:
                        # 處理格式如: "silence_start: 3.240021 | silence_duration: 0.717771"
                        start_part = line.split('silence_start:')[1].strip()
                        start_time = float(start_part.split('|')[0].strip())
                        current_start = start_time
                    except (ValueError, IndexError):
                        continue
                elif 'silence_end:' in line and current_start is not None:
                    try:
                        end_part = line.split('silence_end:')[1].strip()
                        end_time = float(end_part.split('|')[0].strip())
                        silence_segments.append((current_start, end_time))
                        current_start = None
                    except (ValueError, IndexError):
                        current_start = None
            
            print(f"📊 偵測到 {len(silence_segments)} 個空白段落")
            return silence_segments
            
        except Exception as e:
            print(f"⚠️  空白段落偵測失敗: {e}")
            return []
    
    def _remove_silence_segments(self, audio_path, silence_segments, min_gap=0.5):
        """移除空白段落並合併有效音訊"""
        if not silence_segments:
            print("✅ 沒有空白段落需要移除")
            return audio_path
        
        try:
            print(f"✂️  移除空白段落並合併有效音訊...")
            
            # 確保 temp 目錄存在
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(self.temp_dir, f"{filename}_no_silence.wav")
            
            # 使用更簡單的方法：直接使用 silencedetect 的逆向功能
            # 創建一個臨時檔案來儲存非空白段落
            temp_segments = []
            current_time = 0.0
            
            for start, end in silence_segments:
                # 如果當前時間到空白開始之間有足夠的間隔，保存這段
                if start > current_time + min_gap:
                    temp_segments.append((current_time, start))
                current_time = end
            
            # 添加最後一段（如果有的話）
            if current_time < 1000000:  # 假設音訊不會超過這個長度
                temp_segments.append((current_time, 1000000))
            
            if not temp_segments:
                print("⚠️  所有段落都是空白，無法處理")
                return audio_path
            
            # 使用 silencedetect 的逆向功能來移除空白
            # 設定一個很高的閾值，只保留有聲音的部分
            result = subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-af', 'silenceremove=start_periods=1:start_duration=1:start_threshold=-30dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-30dB:detection=peak,aformat=dblp,areverse',
                '-y', output_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # 檢查處理後的檔案大小
                original_size = os.path.getsize(audio_path)
                processed_size = os.path.getsize(output_path)
                reduction_percent = (1 - processed_size / original_size) * 100
                
                print(f"✅ 空白段落移除完成")
                print(f"📊 檔案大小減少: {reduction_percent:.1f}%")
                print(f"📁 處理後檔案: {output_path}")
                
                return output_path
            else:
                print(f"❌ 空白段落移除失敗: {result.stderr}")
                return audio_path
                
        except Exception as e:
            print(f"❌ 空白段落移除失敗: {e}")
            return audio_path
    
    def preprocess_audio(self, audio_path):
        """智能音訊預處理（包含空白段落移除）"""
        print(f"🔧 智能音訊預處理: {audio_path}")
        
        # 分析音訊品質
        audio_info = self._analyze_audio_quality(audio_path)
        if audio_info:
            print(f"   音訊格式: {audio_info['codec']}")
            print(f"   採樣率: {audio_info['sample_rate']} Hz")
            print(f"   聲道數: {audio_info['channels']}")
            print(f"   時長: {audio_info['duration']:.1f} 秒")
            print(f"   音量: {audio_info['volume']:.1f} dB")
            
            # 動態調整音量增強（更保守的設定）
            current_volume = audio_info['volume']
            target_volume = -12  # 更保守的目標音量
            volume_boost = max(0, target_volume - current_volume)
            volume_boost = min(volume_boost, self.optimized_params['volume_boost'])
            
            print(f"   動態音量增強: {volume_boost:.1f} dB")
        else:
            volume_boost = self.optimized_params['volume_boost']
            print(f"   使用預設音量增強: {volume_boost:.1f} dB")
        
        try:
            # 確保 temp 目錄存在
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            temp_wav = os.path.join(self.temp_dir, f"{filename}_temp.wav")
            converted_path = os.path.join(self.temp_dir, f"{filename}_optimized.wav")
            
            # 第一步：轉換為 WAV
            print("   轉換為 WAV 格式...")
            result1 = subprocess.run([
                'ffmpeg', '-i', audio_path, 
                '-y', temp_wav
            ], capture_output=True, text=True)
            
            if result1.returncode != 0:
                print(f"❌ WAV 轉換失敗: {result1.stderr}")
                return audio_path
            
            # 第二步：偵測並移除空白段落（溫和設定）
            silence_segments = self._detect_silence_segments(temp_wav, silence_threshold=-30, min_silence_duration=1.0)
            if silence_segments:
                temp_wav = self._remove_silence_segments(temp_wav, silence_segments)
            
            # 第三步：智能優化格式（溫和處理）
            print("   智能優化音訊格式...")
            # 使用溫和的音量增強和音頻處理，避免過度處理
            # 適度的音量增強，輕微的噪音過濾
            filter_chain = f"volume={volume_boost}dB,highpass=f=100,lowpass=f=7000,afftdn=nf=-20"
            
            result2 = subprocess.run([
                'ffmpeg', '-i', temp_wav, 
                '-af', filter_chain,
                '-ar', '16000',  # 16kHz 採樣率
                '-ac', '1',      # 單聲道
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-y', converted_path
            ], capture_output=True, text=True)
            
            # 清理臨時檔案
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            if result2.returncode == 0:
                print(f"✅ 智能優化完成: {converted_path}")
                return converted_path
            else:
                print(f"❌ 優化失敗: {result2.stderr}")
                print(f"   使用原始檔案繼續處理")
                return audio_path
                
        except Exception as e:
            print(f"❌ 音訊預處理失敗: {e}")
            return audio_path
    
    def load_model(self):
        """智能載入模型（Breeze-ASR-25 + faster-whisper 備用）"""
        print("🤖 智能載入語音轉錄模型...")
        print("🎯 主要模型: Breeze-ASR-25 (台灣中文優化)")
        if FASTER_WHISPER_AVAILABLE:
            print("🔄 備用模型: faster-whisper (高效能 Whisper)")
        else:
            print("🔄 備用模型: transformers Whisper (標準 Whisper)")
        
        # 強制垃圾回收
        gc.collect()
        
        # 使用智能偵測的硬體配置
        device = self.hardware_info['device']
        torch_dtype = self.optimized_params['torch_dtype']
        batch_size = self.optimized_params['batch_size']
        
        print(f"✅ 使用 {self.hardware_info['acceleration']} 加速")
        print(f"✅ 使用 {torch_dtype} 精度")
        print(f"✅ 批次大小: {batch_size}")

        # 載入主要模型 (Breeze-ASR-25)
        try:
            self.model = pipeline(
                task="automatic-speech-recognition",
                model="MediaTek-Research/Breeze-ASR-25",
                device=device,
                torch_dtype=torch_dtype,
                # 智能優化參數
                batch_size=batch_size,
                chunk_length_s=self.optimized_params['chunk_length_s'],
                stride_length_s=self.optimized_params['stride_length_s'],
                return_timestamps=True
            )
            print("✅ Breeze-ASR-25 模型載入完成！")
            self.model_name = "Breeze-ASR-25"
            self.model_type = "transformers"
        except Exception as e:
            print(f"⚠️  Breeze-ASR-25 載入失敗: {e}")
            print("🔄 切換到備用 Whisper 模型...")
            
            # 嘗試使用 faster-whisper
            if FASTER_WHISPER_AVAILABLE:
                try:
                    self._load_faster_whisper_model()
                    self.model_name = "faster-whisper"
                    self.model_type = "faster-whisper"
                except Exception as e2:
                    print(f"⚠️  faster-whisper 載入失敗: {e2}")
                    print("🔄 切換到標準 transformers Whisper...")
                    self._load_standard_whisper_model()
                    self.model_name = "Whisper"
                    self.model_type = "transformers"
            else:
                self._load_standard_whisper_model()
                self.model_name = "Whisper"
                self.model_type = "transformers"
        
        # 檢查載入後記憶體使用
        memory_after = psutil.virtual_memory().percent
        print(f"📊 載入後記憶體使用: {memory_after:.1f}%")
        
        # 記憶體警告
        if memory_after > 90:
            print("⚠️  記憶體使用率過高，建議關閉其他應用程式")
        elif memory_after > 80:
            print("💡 記憶體使用率較高，建議監控系統資源")
    
    def _load_faster_whisper_model(self):
        """載入 faster-whisper 模型"""
        print("🚀 載入 faster-whisper 模型...")
        
        # 智能選擇模型大小和計算類型
        if self.hardware_info['acceleration'] == 'CUDA':
            model_size = "large-v3"
            compute_type = "float16"
        elif self.hardware_info['acceleration'] == 'MPS':
            model_size = "large-v2"  # MPS 對 v3 支援可能不完整
            compute_type = "float16"
        else:
            model_size = "medium"
            compute_type = "int8"
        
        print(f"   模型大小: {model_size}")
        print(f"   計算類型: {compute_type}")
        
        self.model = WhisperModel(
            model_size, 
            device=self.hardware_info['device'], 
            compute_type=compute_type
        )
        print("✅ faster-whisper 模型載入完成！")
    
    def _load_standard_whisper_model(self):
        """載入標準 transformers Whisper 模型"""
        print("🔄 載入標準 transformers Whisper 模型...")
        
        device = self.hardware_info['device']
        torch_dtype = self.optimized_params['torch_dtype']
        
        self.model = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-base",
            device=device,
            torch_dtype=torch_dtype,
            return_timestamps=True
        )
        print("✅ 標準 Whisper 模型載入完成！")
    
    def transcribe_with_fallback(self, audio_path):
        """智能轉錄，如果結果只有驚嘆號則切換到備用模型"""
        # 首先嘗試主要模型
        if self.model_type == "faster-whisper":
            result = self._transcribe_with_faster_whisper(audio_path)
        else:
            result = self.model(audio_path, return_timestamps=True)
        
        # 檢查結果是否只有驚嘆號
        text = result['text'].strip()
        if text == '!' or text == '!!!!!!!!!' or len(text) <= 2:
            print("⚠️  主要模型無法識別內容，嘗試備用模型...")
            
            # 載入備用模型
            if self.model_name == "Breeze-ASR-25":
                print("🔄 切換到備用 Whisper 模型...")
                if FASTER_WHISPER_AVAILABLE:
                    try:
                        backup_model = WhisperModel("large-v2", device=self.hardware_info['device'], compute_type="float16")
                        result = self._transcribe_with_faster_whisper(audio_path, backup_model)
                        print("✅ 使用 faster-whisper 模型重新轉錄")
                    except:
                        backup_model = pipeline(
                            task="automatic-speech-recognition",
                            model="openai/whisper-base",
                            device=self.hardware_info['device'],
                            torch_dtype=self.optimized_params['torch_dtype'],
                            return_timestamps=True
                        )
                        result = backup_model(audio_path, return_timestamps=True)
                        print("✅ 使用標準 Whisper 模型重新轉錄")
                else:
                    backup_model = pipeline(
                        task="automatic-speech-recognition",
                        model="openai/whisper-base",
                        device=self.hardware_info['device'],
                        torch_dtype=self.optimized_params['torch_dtype'],
                        return_timestamps=True
                    )
                    result = backup_model(audio_path, return_timestamps=True)
                    print("✅ 使用標準 Whisper 模型重新轉錄")
            else:
                print("⚠️  備用模型也無法識別，返回原始結果")
        
        return result
    
    def _transcribe_with_faster_whisper(self, audio_path, model=None):
        """使用 faster-whisper 進行轉錄"""
        if model is None:
            model = self.model
        
        # 使用 faster-whisper 的 VAD 和優化參數
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # 轉換為標準格式
        full_text = "".join(segment.text for segment in segments)
        
        # 創建標準格式的結果
        result = {
            "text": full_text,
            "chunks": []
        }
        
        # 重新獲取帶時間戳的結果
        segments_with_timestamps, _ = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        for segment in segments_with_timestamps:
            result["chunks"].append({
                "text": segment.text,
                "timestamp": [segment.start, segment.end]
            })
        
        return result
        
    def find_audio_files(self):
        """自動偵測音訊檔案"""
        audio_files = []
        for format in self.supported_formats:
            files = glob.glob(f"source{format}")
            audio_files.extend(files)
        return audio_files
    
    def get_file_info(self, file_path):
        """獲取檔案資訊"""
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024*1024)
        
        # 使用 ffprobe 獲取準確的時長
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', file_path
            ], capture_output=True, text=True, check=True)
            
            import json
            info = json.loads(result.stdout)
            duration_seconds = float(info['format']['duration'])
            duration_minutes = duration_seconds / 60
        except:
            # 如果無法獲取時長，使用檔案大小估算
            duration_minutes = file_size_mb
        
        estimated_chunks = max(1, int(duration_minutes * 60 / self.optimized_params['segment_duration']))
        
        return {
            'size_mb': file_size_mb,
            'duration_min': duration_minutes,
            'chunks': estimated_chunks
        }
    
    def check_existing_transcription(self, audio_path, processed_audio=None, processed_file_info=None):
        """檢查是否已有轉錄結果"""
        if not os.path.exists(self.output_dir):
            return None
        
        # 獲取音訊檔案的基本資訊
        file_size = os.path.getsize(audio_path)
        file_size_mb = file_size / (1024*1024)
        
        # 獲取音訊長度
        file_info = self.get_file_info(audio_path)
        duration_min = file_info['duration_min']
        
        print(f"🔍 檢查現有轉錄結果...")
        print(f"📊 當前音檔: {file_size_mb:.1f} MB, {duration_min:.1f} 分鐘")
        
        # 如果沒有提供處理後的檔案資訊，則進行預處理
        if processed_audio is None or processed_file_info is None:
            processed_audio = self.preprocess_audio(audio_path)
            processed_file_info = self.get_file_info(processed_audio)
        
        processed_size_mb = processed_file_info['size_mb']
        processed_duration_min = processed_file_info['duration_min']
        
        print(f"📊 處理後音檔: {processed_size_mb:.1f} MB, {processed_duration_min:.1f} 分鐘")
        
        # 尋找可能的轉錄結果檔案
        result_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith("result-") and file.endswith(".txt"):
                result_files.append(file)
        
        # 按檔名中的時間戳排序，最新的在前
        def extract_timestamp(filename):
            # 從檔名中提取時間戳，格式如：result-source-20250903_220356.txt
            try:
                parts = filename.split('-')
                if len(parts) >= 3:
                    timestamp_str = parts[-1].replace('.txt', '')
                    # 轉換為 datetime 物件進行比較
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                return datetime.min
            except:
                return datetime.min
        
        result_files.sort(key=extract_timestamp, reverse=True)
        
        for result_file in result_files:
            result_path = os.path.join(self.output_dir, result_file)
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # 檢查檔案大小和音訊長度是否匹配（使用處理後的檔案資訊）
                size_match = f"檔案大小: {processed_size_mb:.1f} MB" in content
                duration_match = f"音訊長度: {processed_duration_min:.1f} 分鐘" in content
                
                if size_match and duration_match:
                    print(f"✅ 找到匹配的轉錄結果: {result_file}")
                    print(f"📊 檔案大小: {processed_size_mb:.1f} MB")
                    print(f"⏱️  音訊長度: {processed_duration_min:.1f} 分鐘")
                    
                    # 檢查是否有實際的轉錄內容
                    has_content = False
                    first_sentence = None
                    
                    # 提取第一句話來驗證
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        # 跳過標題、分隔線、檔案資訊等
                        if (line and 
                            not line.startswith('=') and 
                            not line.startswith('檔案:') and 
                            not line.startswith('模型:') and 
                            not line.startswith('處理') and 
                            not line.startswith('語音轉錄結果') and 
                            not line.startswith('分段轉錄結果') and 
                            not line.startswith('分段時間戳') and
                            not line.startswith('檔案大小:') and
                            not line.startswith('音訊長度:') and
                            not line.startswith('轉錄時間:') and
                            not line.startswith('分塊大小:') and
                            not line.startswith('記憶體使用率:') and
                            not line.startswith('--- 續轉結果') and
                            len(line) > 20):  # 確保是實際的轉錄內容
                            
                            # 檢查是否為時間戳格式
                            if not (line.startswith('[') and ']' in line):
                                first_sentence = line
                                has_content = True
                                print(f"📝 第一句話: {first_sentence[:50]}...")
                                break
                    
                    if has_content:
                        print(f"✅ 找到完整的轉錄結果，將續接此檔案")
                        return result_path, first_sentence, processed_audio, processed_file_info
                    else:
                        print(f"⚠️  檔案匹配但無轉錄內容，將續接此檔案")
                        return result_path, "無內容", processed_audio, processed_file_info
                else:
                    if not size_match:
                        print(f"⚠️  檔案大小不匹配: {result_file}")
                    if not duration_match:
                        print(f"⚠️  音訊長度不匹配: {result_file}")
            except Exception as e:
                continue
        
        print(f"❌ 未找到匹配的轉錄結果，將創建新檔案")
        return None
    
    def transcribe_audio_segments(self, audio_path, segment_duration=300):
        """分段轉錄音訊檔案"""
        print(f"\n🎵 正在分段處理音訊檔案: {audio_path}")
        
        # 預處理音訊
        processed_audio = self.preprocess_audio(audio_path)
        
        # 獲取檔案資訊
        file_info = self.get_file_info(processed_audio)
        print(f"📊 檔案大小: {file_info['size_mb']:.1f} MB")
        print(f"⏱️  音訊長度: {file_info['duration_min']:.1f} 分鐘")
        
        # 計算分段數
        total_duration = file_info['duration_min'] * 60  # 轉為秒
        num_segments = int(total_duration / segment_duration) + 1
        print(f"🔢 將分為 {num_segments} 個 {segment_duration} 秒的段落")
        
        # 檢查是否已有轉錄結果（傳遞處理後的檔案資訊）
        existing_result = self.check_existing_transcription(audio_path, processed_audio, file_info)
        if existing_result:
            result_path, first_sentence, processed_audio, file_info = existing_result
            print(f"✅ 發現現有轉錄結果，將進行續轉處理")
            return self.create_segmented_transcription(audio_path, processed_audio, file_info, segment_duration, result_path)
        
        # 創建分段轉錄結果
        return self.create_segmented_transcription(audio_path, processed_audio, file_info, segment_duration)
    
    def create_segmented_transcription(self, audio_path, processed_audio, file_info, segment_duration, existing_result_path=None):
        """創建分段轉錄或續接現有結果"""
        if existing_result_path and os.path.exists(existing_result_path):
            # 續接現有結果檔
            print(f"📁 續接現有結果檔案: {existing_result_path}")
            return existing_result_path, file_info
        else:
            # 創建新的結果檔
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_file = f"{self.output_dir}/result-{filename}-{timestamp}.txt"
            
            # 創建輸出目錄
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # 初始化結果檔案
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("智能語音轉錄結果 (Breeze-ASR-25)\n")
                f.write("=" * 60 + "\n")
                f.write(f"檔案: {audio_path}\n")
                f.write(f"模型: MediaTek-Research/Breeze-ASR-25\n")
                f.write(f"處理方法: 智能分段轉錄\n")
                f.write(f"硬體配置: {self.hardware_info['description']}\n")
                f.write(f"加速方式: {self.hardware_info['acceleration']}\n")
                f.write(f"記憶體: {self.hardware_info['memory_gb']:.1f} GB\n")
                f.write(f"檔案大小: {file_info['size_mb']:.1f} MB\n")
                f.write(f"音訊長度: {file_info['duration_min']:.1f} 分鐘\n")
                f.write(f"智能分段大小: {self.optimized_params['segment_duration']} 秒\n")
                f.write(f"批次大小: {self.optimized_params['batch_size']}\n")
                f.write(f"精度: {self.optimized_params['torch_dtype']}\n")
                f.write(f"轉錄時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write("智能分段轉錄結果:\n")
                f.write("=" * 60 + "\n")
            
            print(f"📁 新結果檔案已創建: {output_file}")
            return output_file, file_info
    
    def resume_transcription(self, audio_path, processed_audio, file_info, existing_result_path, first_sentence, segment_duration):
        """續轉現有轉錄"""
        print(f"🔄 正在續轉現有轉錄結果...")
        
        # 讀取現有結果
        with open(existing_result_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
        
        # 提取已轉錄的內容
        if "分段轉錄結果:" in existing_content:
            transcribed_content = existing_content.split("分段轉錄結果:")[1]
        else:
            transcribed_content = existing_content
        
        print(f"📝 現有轉錄內容長度: {len(transcribed_content)} 字元")
        print(f"🔍 第一句話: {first_sentence[:50]}...")
        
        # 檢查最後一段的時間戳記，判斷是否完成轉錄
        lines = transcribed_content.strip().split('\n')
        last_timestamp = None
        
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('[') and ']' in line:
                # 提取時間戳記，格式如：[780.0s - 780.0s]
                try:
                    timestamp_part = line.split(']')[0][1:]  # 移除 [ 和 ]
                    if ' - ' in timestamp_part:
                        end_time_str = timestamp_part.split(' - ')[1]
                        if end_time_str.endswith('s'):
                            end_time = float(end_time_str[:-1])
                            last_timestamp = end_time
                            break
                except:
                    continue
        
        # 獲取音訊總長度
        total_duration = file_info['duration_min'] * 60  # 轉為秒
        
        if last_timestamp:
            print(f"📊 最後轉錄時間: {last_timestamp:.1f}s / {total_duration:.1f}s")
        else:
            print(f"📊 最後轉錄時間: 無法解析 / {total_duration:.1f}s")
        
        # 檢查是否已經完成轉錄（允許 30 秒的誤差）
        if last_timestamp and last_timestamp >= (total_duration - 30):
            print(f"✅ 轉錄結果已完整，跳過重新轉錄")
            print(f"📁 現有結果檔案: {existing_result_path}")
            return existing_result_path, file_info, True  # 返回 True 表示已完成
        else:
            if last_timestamp:
                progress = last_timestamp/total_duration*100
                print(f"🔄 轉錄未完成，需要續轉 (進度: {progress:.1f}%)")
            else:
                print(f"🔄 轉錄未完成，需要續轉 (無法解析時間戳記)")
            return existing_result_path, file_info, False  # 返回 False 表示需要續轉
    
    def transcribe_with_realtime_save(self, audio_path, output_file, file_info):
        """分段轉錄並實時保存結果"""
        print(f"🔄 開始分段轉錄並實時保存...")
        
        # 使用優化參數進行轉錄
        result = self.model(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,  # 使用官方建議的 30 秒分塊
            stride_length_s=5,  # 使用官方建議的 5 秒重疊
            batch_size=1
        )
        
        # 實時寫入結果
        self.save_result_realtime(result, output_file)
        
        return result
    
    def transcribe_audio_segments_realtime(self, audio_path, output_file, file_info):
        """智能分段轉錄音訊檔案並實時保存"""
        print(f"🔄 開始智能分段轉錄並實時保存...")
        
        # 使用智能優化的分段大小
        segment_duration = self.optimized_params['segment_duration']
        stride_duration = self.optimized_params['stride_duration']
        
        # 計算分段數
        total_duration = file_info['duration_min'] * 60  # 轉為秒
        num_segments = int(total_duration / segment_duration) + 1
        
        print(f"🔢 智能分段: {num_segments} 個段落，每段 {segment_duration} 秒")
        print(f"📊 重疊時間: {stride_duration} 秒")
        
        # 設置進度監控變數
        self.total_segments = num_segments
        self.current_segment = 0
        
        all_results = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            print(f"📊 處理段落 {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
            
            # 更新進度監控
            self.current_segment = i + 1
            
            try:
                # 使用 ffmpeg 提取音訊段落
                segment_file = os.path.join(self.temp_dir, f"segment_{i}.wav")
                cmd = [
                    'ffmpeg', '-i', audio_path, 
                    '-ss', str(start_time), 
                    '-t', str(end_time - start_time),
                    '-ar', '16000',  # 16kHz 採樣率
                    '-ac', '1',      # 單聲道
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-y', segment_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️  段落 {i+1} 提取失敗，跳過")
                    continue
                
                # 智能轉錄段落 - 使用備用機制
                segment_result = self.transcribe_with_fallback(segment_file)
                
                # 調試：檢查轉錄結果
                print(f"🔍 段落 {i+1} 轉錄結果: {segment_result}")
                
                # 調整時間戳
                if "chunks" in segment_result and segment_result["chunks"]:
                    for chunk in segment_result["chunks"]:
                        if chunk.get('timestamp') and len(chunk['timestamp']) >= 2:
                            # 確保時間戳不為 None
                            if chunk['timestamp'][0] is not None and chunk['timestamp'][1] is not None:
                                chunk['timestamp'] = [
                                    chunk['timestamp'][0] + start_time,
                                    chunk['timestamp'][1] + start_time
                                ]
                            else:
                                # 如果時間戳為 None，設定為段落時間範圍
                                chunk['timestamp'] = [start_time, start_time + (end_time - start_time)]
                
                # 實時寫入結果
                self.save_result_realtime(segment_result, output_file)
                print(f"✅ 段落 {i+1} 轉錄完成並保存")
                
                # 清理臨時檔案
                if os.path.exists(segment_file):
                    os.remove(segment_file)
                
                all_results.append(segment_result)
                
                # 記憶體監控
                current_memory = psutil.virtual_memory().percent
                if current_memory > 90:
                    print(f"⚠️  記憶體使用率: {current_memory:.1f}%，強制垃圾回收")
                    gc.collect()
                
            except Exception as e:
                print(f"❌ 段落 {i+1} 處理失敗: {str(e)}")
                # 清理臨時檔案
                if os.path.exists(segment_file):
                    os.remove(segment_file)
                continue
        
        # 合併所有結果
        combined_result = self.combine_results(all_results)
        return combined_result
    
    def combine_results(self, results):
        """合併多個轉錄結果"""
        combined_chunks = []
        combined_text = ""
        
        for result in results:
            if "chunks" in result and result["chunks"]:
                combined_chunks.extend(result["chunks"])
            if "text" in result and result["text"]:
                combined_text += result["text"] + " "
        
        return {
            "text": combined_text.strip(),
            "chunks": combined_chunks
        }
    
    def cleanup_temp_files(self):
        """清理臨時檔案"""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"🗑️  已清理臨時檔案目錄: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  清理臨時檔案時發生錯誤: {str(e)}")
    
    def _filter_repetitive_content(self, text):
        """過濾重複內容（更嚴格的過濾）"""
        if not text or len(text.strip()) < 3:
            return text
        
        # 檢查是否為重複的單字
        words = text.strip().split()
        if len(words) == 1:
            # 單字重複檢查
            word = words[0]
            if len(word) == 1 and word in ['好', 'A', '啊', '嗯', '哦', '呃', '嗯嗯', '哈哈', '呵']:
                return ""  # 過濾掉重複的單字
        
        # 檢查是否為重複模式（更嚴格）
        if len(words) >= 2:
            # 檢查前2個字是否重複
            first_word = words[0]
            if all(word == first_word for word in words[:2]):
                return ""  # 過濾掉重複模式
        
        # 檢查是否為連續重複字符
        if len(text) > 5:
            # 檢查是否有超過3個連續相同字符
            for i in range(len(text) - 3):
                if text[i] == text[i+1] == text[i+2] == text[i+3]:
                    return ""  # 過濾掉連續重複字符
        
        return text
    
    def save_result_realtime(self, result, output_file):
        """實時保存轉錄結果（過濾重複內容）"""
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                # 安全處理時間戳
                if "chunks" in result and result["chunks"]:
                    print(f"🔍 找到 {len(result['chunks'])} 個 chunks")
                    for i, chunk in enumerate(result["chunks"]):
                        try:
                            if chunk.get('timestamp') and len(chunk['timestamp']) >= 2:
                                start_time = chunk['timestamp'][0]
                                end_time = chunk['timestamp'][1]
                                text = chunk.get('text', '')
                                
                                # 過濾重複內容
                                filtered_text = self._filter_repetitive_content(text)
                                if not filtered_text:
                                    print(f"🚫 過濾掉重複內容: {text}")
                                    continue
                                
                                if filtered_text.strip():
                                    # 確保時間戳不為 None
                                    if start_time is not None and end_time is not None:
                                        f.write(f"[{start_time:.1f}s - {end_time:.1f}s] {filtered_text}\n")
                                    else:
                                        f.write(f"[時間戳未知] {filtered_text}\n")
                                    f.flush()  # 強制寫入檔案
                                    print(f"✅ 已保存 chunk {i+1}: {filtered_text}")
                        except Exception as e:
                            text = chunk.get('text', '')
                            filtered_text = self._filter_repetitive_content(text)
                            if filtered_text.strip():
                                f.write(f"[時間戳錯誤] {filtered_text}\n")
                                f.flush()
                                print(f"✅ 已保存 chunk {i+1} (時間戳錯誤): {filtered_text}")
                else:
                    # 如果沒有 chunks，寫入 text
                    if "text" in result and result["text"]:
                        filtered_text = self._filter_repetitive_content(result["text"])
                        if filtered_text.strip():
                            f.write(filtered_text + "\n")
                            f.flush()
                            print(f"✅ 已保存完整文字: {filtered_text}")
                        else:
                            print("🚫 過濾掉重複內容")
                    else:
                        print("⚠️  沒有找到可保存的文字內容")
            
            print(f"✅ 轉錄結果已實時保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 實時保存結果時發生錯誤: {str(e)}")
            raise e
    
    def transcribe_audio(self, audio_path):
        """智能轉錄音訊檔案（主入口）"""
        print(f"\n🎵 正在智能處理音訊檔案: {audio_path}")
        
        # 智能預處理音訊
        processed_audio = self.preprocess_audio(audio_path)
        
        # 獲取檔案資訊
        file_info = self.get_file_info(processed_audio)
        print(f"📊 檔案大小: {file_info['size_mb']:.1f} MB")
        print(f"⏱️  音訊長度: {file_info['duration_min']:.1f} 分鐘")
        
        # 智能分段轉錄
        output_file, file_info = self.transcribe_audio_segments(audio_path)
        
        # 智能預估處理時間
        segment_duration = self.optimized_params['segment_duration']
        estimated_time = (file_info['duration_min'] * 60 / segment_duration) * 2.0  # 每段約2分鐘處理時間
        self.estimated_duration_minutes = estimated_time
        
        print(f"⏰ 智能預估處理時間: {estimated_time:.1f} 分鐘")
        print("🔄 正在智能處理中，請勿中斷程式...")
        print("💡 提示：Breeze-ASR-25 對台灣中文口音辨識效果最佳")
        print("🤖 智能硬體優化已啟用，自動調整參數以獲得最佳效能")
        
        start_time = time.time()
        
        try:
            # 智能進度監控
            print("⏱️  開始智能轉錄...")
            print("📊 智能進度監控已啟動，每 30 秒會顯示處理狀態和進度百分比")
            print("💡 如果處理時間超過預估時間，可以按 Ctrl+C 中斷重試")
            
            # 開始進度監控
            self.start_progress_monitor()
            
            # 執行智能分段轉錄 - 實時寫入結果
            result = self.transcribe_audio_segments_realtime(
                processed_audio, 
                output_file, 
                file_info
            )
            
            # 停止進度監控
            self.stop_progress_monitor()
            
            # 計算處理時間
            total_time = time.time() - start_time
            print(f"\n✅ 智能轉錄完成！總耗時: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
            
            # 清理轉換的檔案
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.remove(processed_audio)
                print(f"🗑️  已清理轉換檔案: {processed_audio}")
            
            # 清理所有臨時檔案
            self.cleanup_temp_files()
            
            return result, total_time, file_info
            
        except Exception as e:
            # 停止進度監控
            self.stop_progress_monitor()
            print(f"❌ 智能轉錄失敗: {str(e)}")
            # 清理轉換的檔案
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.remove(processed_audio)
            
            # 清理所有臨時檔案
            self.cleanup_temp_files()
            
            # 強制垃圾回收
            gc.collect()
            raise e
    
    def save_result(self, result, total_time, file_info, audio_path, output_file=None):
        """保存轉錄結果（支援分段寫入）"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 如果沒有指定輸出檔案，創建新的
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_file = f"{self.output_dir}/result-{filename}-{timestamp}.txt"
        
        print(f"\n💾 正在保存結果到檔案...")
        
        # 檢查檔案是否已存在（分段轉錄）
        file_exists = os.path.exists(output_file)
        
        if not file_exists:
            # 創建新檔案
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("語音轉錄結果 (Breeze-ASR-25 分段版)\n")
                f.write("=" * 50 + "\n")
                f.write(f"檔案: {audio_path}\n")
                f.write(f"模型: MediaTek-Research/Breeze-ASR-25\n")
                f.write(f"處理方法: 分段轉錄\n")
                f.write(f"處理時間: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)\n")
                f.write(f"檔案大小: {file_info['size_mb']:.1f} MB\n")
                f.write(f"音訊長度: {file_info['duration_min']:.1f} 分鐘\n")
                f.write(f"智能分段大小: {self.optimized_params['segment_duration']}秒, 重疊: {self.optimized_params['stride_duration']}秒\n")
                f.write(f"記憶體使用率: {self.hardware_info['memory_percent']:.1f}%\n")
                f.write(f"轉錄時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write("分段轉錄結果:\n")
                f.write("=" * 50 + "\n")
        else:
            # 追加到現有檔案
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- 續轉結果 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        
        # 寫入轉錄內容
        with open(output_file, "a", encoding="utf-8") as f:
            # 安全處理時間戳
            if "chunks" in result and result["chunks"]:
                for i, chunk in enumerate(result["chunks"]):
                    try:
                        if chunk.get('timestamp') and len(chunk['timestamp']) >= 2:
                            start_time = chunk['timestamp'][0]
                            end_time = chunk['timestamp'][1]
                            text = chunk.get('text', '')
                            f.write(f"[{start_time:.1f}s - {end_time:.1f}s] {text}\n")
                        else:
                            text = chunk.get('text', '')
                            f.write(f"[時間戳未知] {text}\n")
                    except Exception as e:
                        text = chunk.get('text', '')
                        f.write(f"[時間戳錯誤] {text}\n")
            else:
                # 如果沒有 chunks，寫入 text
                if "text" in result and result["text"]:
                    f.write(result["text"])
        
        print(f"✅ 結果已保存到: {output_file}")
        return output_file
    
    def display_result(self, result):
        """顯示轉錄結果"""
        print("=" * 50)
        print("轉錄結果:")
        print(result["text"])
        
        # 安全處理時間戳顯示
        if "chunks" in result and result["chunks"]:
            print("\n" + "=" * 50)
            print("分段時間戳:")
            for i, chunk in enumerate(result["chunks"]):
                try:
                    if chunk.get('timestamp') and len(chunk['timestamp']) >= 2:
                        start_time = chunk['timestamp'][0]
                        end_time = chunk['timestamp'][1]
                        text = chunk.get('text', '')
                        print(f"[{start_time:.1f}s - {end_time:.1f}s] {text}")
                    else:
                        text = chunk.get('text', '')
                        print(f"[時間戳未知] {text}")
                except Exception as e:
                    text = chunk.get('text', '')
                    print(f"[時間戳錯誤] {text}")
    
    def run(self):
        """智能主執行函數"""
        print("🤖 智能語音轉錄工具 - 基於 Breeze-ASR-25")
        print("=" * 60)
        print("🎯 針對台灣中文口音優化")
        print("🔧 智能硬體偵測和自動參數優化")
        print("💡 保持最佳台灣中文辨識效果")
        print("🤖 無需手動設定參數，全自動智能優化")
        
        # 智能載入模型
        self.load_model()
        
        # 自動偵測音訊檔案
        audio_files = self.find_audio_files()
        
        if not audio_files:
            print("❌ 未找到音訊檔案！")
            print(f"請將音訊檔案命名為: source.aac, source.mp3, source.wav 等")
            return
        
        print(f"\n🔍 自動偵測到 {len(audio_files)} 個音訊檔案:")
        for file in audio_files:
            print(f"  - {file}")
        
        # 智能處理每個音訊檔案
        for audio_file in audio_files:
            try:
                print(f"\n{'='*60}")
                print(f"🎵 開始智能處理: {audio_file}")
                print(f"{'='*60}")
                
                # 智能檢查是否已有轉錄結果
                existing_result = self.check_existing_transcription(audio_file)
                if existing_result:
                    if len(existing_result) == 4:
                        result_path, first_sentence, processed_audio, file_info = existing_result
                    else:
                        result_path, first_sentence = existing_result
                        processed_audio = None
                        file_info = None
                    print(f"✅ 發現現有轉錄結果: {result_path}")
                    print(f"📝 第一句話: {first_sentence[:50]}...")
                    
                    # 智能檢查是否需要續轉
                    if processed_audio and file_info:
                        resume_result = self.resume_transcription(audio_file, processed_audio, file_info, result_path, first_sentence, self.optimized_params['segment_duration'])
                    else:
                        # 如果沒有處理後的檔案資訊，先進行預處理
                        processed_audio = self.preprocess_audio(audio_file)
                        file_info = self.get_file_info(processed_audio)
                        resume_result = self.resume_transcription(audio_file, processed_audio, file_info, result_path, first_sentence, self.optimized_params['segment_duration'])
                    if len(resume_result) == 3 and resume_result[2]:  # 已完成
                        print(f"🎉 轉錄已完成，跳過處理")
                        continue
                    else:
                        print(f"🔄 將進行智能續轉處理...")
                        # 智能續轉處理 - 直接使用現有結果檔
                        result, total_time, file_info = self.transcribe_audio(audio_file)
                        # 不需要再次保存，因為已經在實時保存中處理了
                        output_file = result_path
                else:
                    # 智能全新轉錄
                    result, total_time, file_info = self.transcribe_audio(audio_file)
                    # 不需要再次保存，因為已經在實時保存中處理了
                    output_file = "已實時保存"
                
                # 顯示結果
                self.display_result(result)
                
                print(f"\n🎉 檔案 {audio_file} 智能轉錄完成！")
                print(f"📁 結果保存在: {output_file}")
                
                # 智能垃圾回收
                gc.collect()
                
            except Exception as e:
                print(f"❌ 智能處理檔案 {audio_file} 時發生錯誤: {str(e)}")
                # 智能垃圾回收
                gc.collect()
                continue
        
        print(f"\n🎯 所有音訊檔案智能處理完成！")

def test_audio_segment(audio_path, start_time=10, duration=10):
    """測試音訊段落轉錄效果"""
    print(f"🧪 測試音訊段落轉錄: {start_time}s - {start_time + duration}s")
    
    transcriber = SmartTranscriber()
    transcriber.load_model()
    
    # 提取測試段落
    test_file = f"test_segment_{start_time}s.wav"
    result = subprocess.run([
        'ffmpeg', '-i', audio_path, 
        '-ss', str(start_time), 
        '-t', str(duration),
        '-y', test_file
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 段落提取失敗: {result.stderr}")
        return
    
    # 預處理音訊
    processed_audio = transcriber.preprocess_audio(test_file)
    
    # 轉錄測試
    try:
        transcription_result = transcriber.transcribe_with_fallback(processed_audio)
        print(f"✅ 轉錄結果: {repr(transcription_result['text'])}")
        print(f"📊 長度: {len(transcription_result['text'])} 字元")
    except Exception as e:
        print(f"❌ 轉錄失敗: {e}")
    
    # 清理測試檔案
    for file in [test_file, processed_audio]:
        if os.path.exists(file) and file != audio_path:
            os.remove(file)

def main():
    """簡化的主函數 - 無需參數"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 測試模式
        test_audio_segment("source.aac", start_time=12, duration=10)
    else:
        # 正常模式
        print("🚀 啟動智能語音轉錄工具...")
        transcriber = SmartTranscriber()
        transcriber.run()

if __name__ == "__main__":
    main()
