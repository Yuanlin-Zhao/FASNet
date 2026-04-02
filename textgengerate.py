import os
import numpy as np
from PIL import Image


def is_black_image(img_path):
    """
    判断图像是否全黑（所有像素为0）
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)

        # 如果是多通道图像，只要所有像素都为0就认为是全黑
        return np.all(img_array == 0)
    except Exception as e:
        print(f"读取标签失败: {img_path}, 错误: {e}")
        return False


def generate_text_descriptions(root_path, description="target"):
    """
    为数据集生成对应的文本描述文件。
    如果对应 label 是全黑图像，则 txt 写入 "none"
    否则写入 description（默认 target）

    目录结构：
    root/
      train/
        images/
        labels/
        texts/
      val/
        images/
        labels/
        texts/
    """
    phases = ['train', 'val']
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')

    for phase in phases:
        phase_dir = os.path.join(root_path, phase)
        img_dir = os.path.join(phase_dir, 'images')
        label_dir = os.path.join(phase_dir, 'labels')
        text_dir = os.path.join(phase_dir, 'texts')

        if not os.path.exists(img_dir):
            print(f"跳过: 未找到目录 {img_dir}")
            continue

        if not os.path.exists(label_dir):
            print(f"跳过: 未找到目录 {label_dir}")
            continue

        if not os.path.exists(text_dir):
            os.makedirs(text_dir)
            print(f"已创建目录: {text_dir}")

        count = 0
        none_count = 0

        for filename in os.listdir(img_dir):
            if filename.lower().endswith(valid_ext):
                stem = os.path.splitext(filename)[0]

                # 找对应的 label 文件
                label_path = None
                for ext in valid_ext:
                    possible_path = os.path.join(label_dir, stem + ext)
                    if os.path.exists(possible_path):
                        label_path = possible_path
                        break

                if label_path is None:
                    print(f"未找到标签文件: {stem}")
                    continue

                # 判断是否全黑
                if is_black_image(label_path):
                    text_content = "none"
                    none_count += 1
                else:
                    text_content = description

                text_file_path = os.path.join(text_dir, f"{stem}.txt")

                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)

                count += 1

        print(f"【{phase}】阶段: 共生成 {count} 个文本文件，其中 none 有 {none_count} 个。")


if __name__ == "__main__":
    dataset_root = r"D:\zyl\IRSeg\irseg\data\dataset\medicine"
    generate_text_descriptions(dataset_root)