# SepFormer with LibriMix

本專案基於 [JorisCos/LibriMix](https://github.com/JorisCos/LibriMix) 實現，用於音頻分離任務。

## 環境設置

1. 創建並激活虛擬環境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 數據集

本專案使用 LibriMix 數據集，您可以從 [JorisCos/LibriMix](https://github.com/JorisCos/LibriMix) 下載。

## 使用方法

1. 訓練模型：
```bash
python train.py
```

2. 評估模型：
```bash
python evaluate.py
```

3. 分離音頻：
```bash
python separate.py
```

## 引用

如果您使用了本專案，請引用：

```bibtex
@misc{libriMix,
  author = {JorisCos},
  title = {LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/JorisCos/LibriMix}
}
``` 