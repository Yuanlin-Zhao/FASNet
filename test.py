import os
import numpy as np
from PIL import Image

from tqdm import tqdm
# -----------------------------
# 2. 路径配置
# -----------------------------
pred_folder = r'D:\aaafenge\medical-segmentation-pytorch-main\predict_result\resunet'
label_folder = r'D:\xianyu\irseg\data\01\medicine\validation\masks'


# -----------------------------
# 1. 指标计算类 (学术标准：全局混淆矩阵)
# -----------------------------
class AverageMeter:
    """用于累加混淆矩阵并计算全局指标"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # 混淆矩阵: 行是 Ground Truth, 列是 Prediction
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0

    def update(self, pred, label):
        """
        pred, label: 2D numpy arrays
        """
        mask = (label >= 0) & (label < self.num_classes)
        # 核心：通过位移计算索引，快速生成混淆矩阵
        label_flat = self.num_classes * label[mask].astype('int') + pred[mask].astype('int')
        count = np.bincount(label_flat, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
        self.total_pixels += label.size

    def get_results(self):
        """计算并返回学术常用指标"""
        cm = self.confusion_matrix
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        # 1. IoU (Intersection over Union)
        # 目标类 IoU
        target_iou = tp / (tp + fp + fn + 1e-8)
        # 平均 IoU (mIoU)
        iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8)
        miou = np.mean(iu)

        # 2. Precision / Recall / F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)  # 等同于像素级 Pd
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # 3. Pd (Probability of Detection) & Fa (False Alarm Rate)
        # 学术界 Pd 通常指 Recall
        Pd = recall
        # 学术界 Fa 定义：误报像素数 / 图像总像素数 (通常展示为 e-6)
        Fa = fp / (self.total_pixels + 1e-8)

        return {
            "mIoU": miou,
            "Target_IoU": target_iou,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Pd": Pd,
            "Fa": Fa
        }




def main():
    meter = AverageMeter(num_classes=2)

    # 获取预测文件列表
    pred_files = [f for f in os.listdir(pred_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 建立 label 映射（加速查找）
    label_dict = {os.path.splitext(f)[0]: f for f in os.listdir(label_folder)}

    matched_count = 0
    print(f"Starting evaluation on {len(pred_files)} files...")

    for f in tqdm(pred_files):
        name = os.path.splitext(f)[0]
        if name not in label_dict:
            continue

        pred_path = os.path.join(pred_folder, f)
        label_path = os.path.join(label_folder, label_dict[name])

        # 1. 读取图像
        pred_pil = Image.open(pred_path).convert('L')
        label_pil = Image.open(label_path).convert('L')

        # --- 新增：尺寸对齐检查 ---
        if pred_pil.size != label_pil.size:
            # 注意：PIL 的 size 是 (Width, Height)，与 numpy 的 (H, W) 相反
            # 这里统一将预测图缩放到标签图的大小，使用 NEAREST 保证二值性
            pred_pil = pred_pil.resize(label_pil.size, resample=Image.NEAREST)

        pred_img = np.array(pred_pil)
        label_img = np.array(label_pil)

        # 2. 二值化 (0 or 1)
        pred_bin = (pred_img > 127).astype(np.uint8)
        label_bin = (label_img > 127).astype(np.uint8)

        # 3. 更新全局统计
        meter.update(pred_bin, label_bin)
        matched_count += 1

    # -----------------------------
    # 3. 输出结果
    # -----------------------------
    res = meter.get_results()

    print("\n" + "=" * 40)
    print(f"Evaluation Summary ({matched_count} images)")
    print("=" * 40)
    print(f"mIoU:        {res['mIoU']:.4f}")
    print(f"Target IoU:  {res['Target_IoU']:.4f}")
    print(f"Precision:   {res['Precision']:.4f}")
    print(f"Recall (Pd): {res['Recall']:.4f}")
    print(f"F1-Score:    {res['F1']:.4f}")
    print(f"Fa (10^-6):  {res['Fa'] * 1e6:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()