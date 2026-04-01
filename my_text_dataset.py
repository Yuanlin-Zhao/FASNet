import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer  # 使用 HuggingFace 的 CLIP 工具
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class get_Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, max_tokens=77):
        super(get_Dataset, self).__init__()
        self.flag = "train" if train else "val"
        self.transforms = transforms
        self.max_tokens = max_tokens

        # 修改后 (指向本地路径)
        local_path = r"D:\aaafenge\irseg\clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(local_path)

        # 1. 定义类别与调色板
        self.classes = ('background', 'object')
        self.palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

        # 2. 匹配路径（图像、标签、文本）
        self.data_list = self._load_paired_paths(root)

    def _load_paired_paths(self, root):
        data_list = []
        base_dir = os.path.join(root, self.flag)

        image_dir = os.path.join(base_dir, 'images')
        label_dir = os.path.join(base_dir, 'labels')

        text_dir = os.path.join(base_dir, 'texts')  # 假设文本存放在 texts 文件夹

        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')

        # 建立标签和文本的索引映射
        label_dict = {os.path.splitext(f)[0]: os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.lower().endswith(valid_ext)}

        # 文本通常是 .txt 格式
        text_dict = {os.path.splitext(f)[0]: os.path.join(text_dir, f)
                     for f in os.listdir(text_dir) if f.lower().endswith('.txt')}

        for im in os.listdir(image_dir):
            if im.lower().endswith(valid_ext):
                stem = os.path.splitext(im)[0]
                # 必须同时拥有图像、标签和描述文本
                if stem in label_dict and stem in text_dict:
                    data_list.append({
                        'img': os.path.join(image_dir, im),
                        'lb': label_dict[stem],
                        'txt': text_dict[stem]
                    })

        if len(data_list) == 0:
            print(f"Warning: No paired data found in {base_dir}!")
        return data_list

    def color_to_label(self, mask_array):
        """
        将 RGB 转换成索引图。
        注：如果你的 mask 只有 0 和 1，其实可以直接取单通道，
        这里保留你的欧氏距离逻辑以兼容复杂的调色板。
        """
        mask_3d = mask_array[:, :, np.newaxis, :]
        palette_3d = self.palette[np.newaxis, np.newaxis, :, :]
        diff = np.sum((mask_3d - palette_3d) ** 2, axis=-1)
        label = np.argmin(diff, axis=-1)
        return label.astype(np.uint8)

    def __getitem__(self, idx):
        # 1. 处理图像
        img = Image.open(self.data_list[idx]['img']).convert('RGB')

        # 2. 处理标签
        mask_color = Image.open(self.data_list[idx]['lb']).convert('RGB')
        mask_idx = self.color_to_label(np.array(mask_color))
        mask = Image.fromarray(mask_idx)

        # 3. 处理文本描述
        with open(self.data_list[idx]['txt'], 'r', encoding='utf-8') as f:
            description = f.read().strip()

        # 使用 CLIP Tokenizer 处理文本
        # padding/truncation 保证长度一致 (CLIP 标准长度是 77)
        text_inputs = self.tokenizer(
            description,
            padding='max_length',
            max_length=self.max_tokens,
            truncation=True,
            return_tensors="pt"
        )

        # text_inputs['input_ids'] 形状为 [1, 77]，去掉 batch 维度
        input_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)

        # 4. 变换
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 返回：图像，标签，文本 ID，文本掩码
        return img, mask, input_ids, attention_mask

    def __len__(self):
        return len(self.data_list)