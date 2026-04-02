import os
import time
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from train import parse_args, create_model
from torchvision import transforms
from transformers import AutoTokenizer  # Or whatever tokenizer you used in training




def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    args = parse_args()

    # --- 路径设置 ---
    test_path = r'D:\zyl\IRSeg\irseg\data\dataset\medicine\val\images'
    # 标签路径 (用于计算指标)
    label_path = r'D:\zyl\IRSeg\irseg\data\dataset\medicine\val\images'
    # 文本路径 (用于模型输入)
    text_path = r'D:\zyl\IRSeg\irseg\data\dataset\medicine\val\texts'

    save_path = 'predictions/{}'.format(args.model)
    weights_path = "save_weights/{}/best_model.pth".format(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = create_model(args).to(device)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.eval()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 图像预处理
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.Resize([args.train_size, args.train_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    print(f"Starting inference on {len(os.listdir(test_path))} images...")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\zyl\IRSeg\irseg\clip-vit-base-patch32", local_files_only=True)
    with torch.no_grad():
        for i in tqdm(os.listdir(test_path)):

            img_full_path = os.path.join(test_path, i)
            original_img = Image.open(img_full_path).convert('RGB')
            ori_w, ori_h = original_img.size
            img_tensor = data_transform(original_img).unsqueeze(0).to(device)


            txt_file_name = os.path.splitext(i)[0] + ".txt"
            txt_full_path = os.path.join(text_path, txt_file_name)

            with open(txt_full_path, 'r', encoding='utf-8') as f:
                text_data = f.read().strip()


            text_inputs = tokenizer(
                text_data,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)


            output, _ = model(
                img_tensor,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )


            output = F.interpolate(output, (ori_h, ori_w), mode='bilinear', align_corners=True)
            prediction = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


            gt_full_path = os.path.join(label_path, i)
            if os.path.exists(gt_full_path):
                gt_img = Image.open(gt_full_path).convert('L')
                gt_array = (np.array(gt_img) > 0).astype(np.uint8)



            save_mask = (prediction * 255).astype(np.uint8)
            Image.fromarray(save_mask).save(os.path.join(save_path, i))




if __name__ == '__main__':
    main()