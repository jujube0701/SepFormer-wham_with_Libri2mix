import torch
import torchaudio
import numpy as np
import os
from pesq import pesq
import mir_eval
import warnings
from speechbrain.inference.separation import SepformerSeparation
from scipy.optimize import linear_sum_assignment

# 忽略 mir_eval 的棄用警告
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_metrics(clean_sources, enhanced_sources, sr=8000):
    """計算 SDR、SIR、SAR 和 PESQ 指標"""
    # 轉換為 numpy 數組並確保形狀一致
    clean_sources = [s.cpu().numpy().squeeze() for s in clean_sources]  # [T]
    enhanced_sources = [s.cpu().numpy().squeeze() for s in enhanced_sources]  # [T]
    
    # 確保音頻長度一致
    min_len = min(min(s.shape[0] for s in clean_sources), min(s.shape[0] for s in enhanced_sources))
    clean_sources = [s[:min_len] for s in clean_sources]
    enhanced_sources = [s[:min_len] for s in enhanced_sources]
    
    # 計算每個增強音頻與每個原始音頻的 SDR
    sdr_matrix = np.zeros((len(clean_sources), len(enhanced_sources)))
    for i, clean in enumerate(clean_sources):
        for j, enhanced in enumerate(enhanced_sources):
            sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
                clean[np.newaxis, :],
                enhanced[np.newaxis, :]
            )
            sdr_matrix[i, j] = sdr[0]
    
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-sdr_matrix)
    
    # 重新排序增強音頻以匹配原始音頻
    enhanced_sources = [enhanced_sources[i] for i in col_ind]
    
    # 使用最佳匹配計算最終指標
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
        np.array(clean_sources),
        np.array(enhanced_sources)
    )
    
    # 計算 PESQ
    pesq_scores = []
    for clean, enhanced in zip(clean_sources, enhanced_sources):
        # 確保音頻長度至少為 1/4 秒
        min_length = sr // 4
        if len(clean) < min_length:
            continue
        # 對於 8000Hz 採樣率，使用窄帶模式
        pesq_score = pesq(sr, clean, enhanced, 'nb')
        pesq_scores.append(pesq_score)
    
    return sdr, sir, sar, np.mean(pesq_scores) if pesq_scores else 0.0, col_ind

def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 載入預訓練模型
    print("載入預訓練模型...")
    model = SepformerSeparation.from_hparams(
        source="pretrained_models/sepformer-wham",
        savedir="pretrained_models/sepformer-wham",
        run_opts={"device": device}
    )
    
    # 設置數據目錄
    test_dir = "data/LibriMix/Libri2Mix/wav8k/min/test"
    mix_dir = os.path.join(test_dir, "mix_clean")
    s1_dir = os.path.join(test_dir, "s1")
    s2_dir = os.path.join(test_dir, "s2")
    
    # 初始化指標
    total_sdr = 0
    total_sir = 0
    total_sar = 0
    total_pesq = 0
    count = 0
    
    # 遍歷測試集
    for mix_file in os.listdir(mix_dir):
        if mix_file.endswith(".wav"):
            # 讀取混合音檔和原始音檔
            mix_path = os.path.join(mix_dir, mix_file)
            s1_path = os.path.join(s1_dir, mix_file)
            s2_path = os.path.join(s2_dir, mix_file)
            
            # 讀取音頻
            mix_wav, sr = torchaudio.load(mix_path)
            s1_wav, _ = torchaudio.load(s1_path)
            s2_wav, _ = torchaudio.load(s2_path)
            
            # 確保音頻是單聲道
            if mix_wav.shape[0] > 1:
                mix_wav = mix_wav.mean(dim=0, keepdim=True)
            if s1_wav.shape[0] > 1:
                s1_wav = s1_wav.mean(dim=0, keepdim=True)
            if s2_wav.shape[0] > 1:
                s2_wav = s2_wav.mean(dim=0, keepdim=True)
            
            # 分離音頻
            with torch.no_grad():
                # 使用 separate_batch 方法
                mix_wav = mix_wav.to(device)
                enhanced = model.separate_batch(mix_wav)
            
            # 計算指標並獲取最佳匹配順序
            sdr, sir, sar, pesq_score, col_ind = calculate_metrics(
                [s1_wav, s2_wav],
                [enhanced[:, :, 0], enhanced[:, :, 1]]
            )
            
            # 保存示例音頻
            if count == 0:  # 只保存第一個示例
                # 保存混合音頻
                torchaudio.save(
                    os.path.join("example_audio", "mix.wav"),
                    mix_wav.cpu(),
                    sr
                )
                # 保存原始音頻
                torchaudio.save(
                    os.path.join("example_audio", "s1.wav"),
                    s1_wav,
                    sr
                )
                torchaudio.save(
                    os.path.join("example_audio", "s2.wav"),
                    s2_wav,
                    sr
                )
                # 根據最佳匹配順序保存分離音頻
                torchaudio.save(
                    os.path.join("example_audio", "enhanced1.wav"),
                    enhanced[:, :, col_ind[0]].cpu(),
                    sr
                )
                torchaudio.save(
                    os.path.join("example_audio", "enhanced2.wav"),
                    enhanced[:, :, col_ind[1]].cpu(),
                    sr
                )
            
            # 累加指標
            total_sdr += float(sdr.mean())  # 轉換為 Python float
            total_sir += float(sir.mean())
            total_sar += float(sar.mean())
            total_pesq += float(pesq_score)
            count += 1
            
            # 每處理 10 個文件打印一次平均指標
            if count % 10 == 0:
                print(f"\n處理了 {count} 個文件:")
                print(f"平均 SDR: {total_sdr/count:.2f} dB")
                print(f"平均 SIR: {total_sir/count:.2f} dB")
                print(f"平均 SAR: {total_sar/count:.2f} dB")
                print(f"平均 PESQ: {total_pesq/count:.2f}")
    
    # 打印最終結果
    print("\n最終評估結果:")
    print(f"平均 SDR: {total_sdr/count:.2f} dB")
    print(f"平均 SIR: {total_sir/count:.2f} dB")
    print(f"平均 SAR: {total_sar/count:.2f} dB")
    print(f"平均 PESQ: {total_pesq/count:.2f}")

if __name__ == "__main__":
    main() 