## 📥 Dataset Download

Two datasets are provided for reproducing our experiments:

1. **PAL-style split dataset (for IRSTD, NUDT-SIRST, SIRST3)**  
   - File: `dataset.zip`  
   - Download: [Baidu Netdisk](https://pan.baidu.com/s/12WYuCtyuRBpos4m47_uGJw?pwd=tk8i)  
   - Extraction code: `tk8i`  
   - Note: This dataset follows the **PAL** training/testing split.

2. **KBT dataset (SRSNet-style split)**  
   - File: `KBT.zip`  
   - Download: [Baidu Netdisk](https://pan.baidu.com/s/1mysj2Y92aALRJoYDLgl-rg?pwd=fmmh)  
   - Extraction code: `fmmh`  
   - Note: This dataset follows the **SRSNet** training/testing split.


## 🚀 How to Run

The workflow for reproducing our experiments is as follows:

```bash
# 1. Generate text descriptions
python textgenerate.py

# 2. Train the model
python train.py

# 3. Generate prediction images
python predictiontext.py

# 4. Evaluate metrics
python test.py
```

## 📄 Methods Referenced in This Paper

Our experiments are conducted under consistent data split settings for fair comparison. Specifically, the **IRSTD**, **NUDT-SIRST**, and **SIRST3** benchmarks follow the same training/testing split protocol as **PAL**, while the **KBT** dataset adopts the split strategy defined in **SRSNet**.

### 🔗 Implementations

- **MLCLNet (integrated in MSDA-Net, IPT 2022)**  
  https://github.com/YuChuang1205/MSDA-Net  

- **MSHNet (CVPR 2024)**  
  https://github.com/Lliu666/MSHNet  

- **ConDSeg (AAAI 2025)**  
  https://github.com/Mengqi-Lei/ConDSeg  

- **PAL (ICCV 2025)**  
  https://github.com/YuChuang1205/PAL  

- **UNIP (ICCV 2025)**  
  https://github.com/casiatao/UNIP  

- **SRSNet (TIP 2025)**  
  https://github.com/fidshu/SRSNet  




