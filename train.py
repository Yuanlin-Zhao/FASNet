import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import transforms as T
from torch.cuda.amp import autocast
# 引入混合精度训练库
from torch.cuda.amp import autocast, GradScaler
from train_utils import evaluate, create_lr_scheduler
from my_text_dataset import get_Dataset

from PIL import ImageFile
import random
import numpy as np
from src.IRsegNetText import IRSegNet


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def create_model(args):
    # 你的模型
    model = IRSegNet(in_channels=3, num_classes=args.num_classes)

    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CriterionComposite(nn.Module):
    def __init__(self, num_classes, alpha=0.5):
        super(CriterionComposite, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 平衡因子
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def dice_loss(self, predict, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(predict * predict)
        z_sum = torch.sum(target * target)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        loss_ce = self.ce(inputs, target)
        valid_mask = (target != 255)
        target_dice = target.clone()
        target_dice[~valid_mask] = 0  # 临时填充，后面会mask掉
        target_one_hot = F.one_hot(target_dice, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        inputs_softmax = F.softmax(inputs, dim=1)
        loss_dice = 0.

        for i in range(self.num_classes):

            loss_dice += self.dice_loss(inputs_softmax[:, i] * valid_mask, target_one_hot[:, i] * valid_mask)

        loss_dice = loss_dice / self.num_classes

        return loss_ce + self.alpha * loss_dice


class train_transform:
    def __init__(self, size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [
            T.Resize(size=size),  # 如果显存够，建议换成 RandomResizedCrop
            T.RandomHorizontalFlip(hflip_prob),
            T.RandomVerticalFlip(vflip_prob),

            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class eval_transform:
    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class InfraredAttentionLoss(nn.Module):

    def __init__(
            self,
            lambda_inside=1.0,
            lambda_outside=0.5,
            lambda_bg=0.5,
            lambda_smooth=0.05,
            eps=1e-6
    ):
        super().__init__()

        self.lambda_inside = lambda_inside
        self.lambda_outside = lambda_outside
        self.lambda_bg = lambda_bg
        self.lambda_smooth = lambda_smooth
        self.eps = eps

    def forward(self, att, target):
        p = torch.sigmoid(att)


        inside_loss = (1 - p) * target
        inside_loss = inside_loss.mean()


        dilated = F.max_pool2d(target, kernel_size=7, stride=1, padding=3)

        relax_region = dilated - target

        outside_loss = torch.clamp(p - 0.5, min=0) * relax_region
        outside_loss = outside_loss.mean()


        far_bg = 1 - dilated

        bg_loss = p * far_bg
        bg_loss = bg_loss.mean()

        dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
        dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])

        smooth_loss = dx.mean() + dy.mean()

        loss = (
                self.lambda_inside * inside_loss
                + self.lambda_outside * outside_loss
                + self.lambda_bg * bg_loss
                + self.lambda_smooth * smooth_loss
        )

        return loss


def info_nce_loss(v_feat, t_feat, temperature=0.5):

    v_feat = F.normalize(v_feat, dim=-1)
    t_feat = F.normalize(t_feat, dim=-1)


    pos_sim = torch.sum(v_feat * t_feat, dim=-1)

    loss = F.mse_loss(pos_sim, torch.ones_like(pos_sim))

    return loss * 0.1


def object_alignment_loss(v_feat, t_feat):

    v_feat = F.normalize(v_feat, dim=-1)
    t_feat = F.normalize(t_feat, dim=-1)


    cosine_sim = (v_feat * t_feat).sum(dim=-1)

    loss = (1 - cosine_sim).mean()

    return loss * 0.1


def train_one_epoch_pro(model, optimizer, data_loader, device, epoch, criterion, scaler,
                        lr_scheduler, print_freq=10, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    header = 'Epoch: [{}]'.format(epoch)

    for i, (image, target, input_ids, attention_mask) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with autocast():

            logits, attention_maps, v_feat, t_feat = model(image, input_ids, attention_mask)


            seg_target = target.long()
            if target.dim() == 3:
                att_target = target.unsqueeze(1).float()
            else:
                att_target = target.float()

            seg_loss = criterion(logits, seg_target)

            # --- 原有注意力损失逻辑 ---
            att_criterion = InfraredAttentionLoss()
            att_loss = 0.0
            for att in attention_maps:
                att = F.interpolate(att, size=att_target.shape[-2:], mode='bilinear', align_corners=True)
                att_loss += att_criterion(att, att_target)
            att_loss /= len(attention_maps)


            align_loss = info_nce_loss(v_feat, t_feat)

            loss = seg_loss + 0.1 * att_loss + 0.1 * align_loss

            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

        loss_val = loss.item() * accumulation_steps
        total_loss += loss_val

        if i % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{header} Step [{i}/{len(data_loader)}] Loss: {loss_val:.4f} LR: {lr:.6f}")

    return total_loss / len(data_loader)


import train_utils.utils as utils


def textevaluate(model, data_loader, device, num_classes, criterion):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes=num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for image, target, input_ids, attention_mask in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device)
            target = target.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            output, _ = model(image, input_ids, attention_mask)

            val_loss = criterion(output, target)

            # 更新日志记录
            metric_logger.update(loss=val_loss.item())

            # 更新指标
            pred_label = output.argmax(1)
            confmat.update(target.flatten(), pred_label.flatten())

            if num_classes == 2:
                pred = torch.sigmoid(output[:, 1])
                pred = (pred > 0.5).float()
                target_bin = (target == 1).float()

                intersection = (pred * target_bin).sum(dim=(1, 2))
                union = pred.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2))
                dice = (2. * intersection + 1e-6) / (union + 1e-6)

                total_dice += dice.mean().item()
                total_samples += 1

    confmat.reduce_from_all_processes()
    avg_dice = total_dice / total_samples if total_samples > 0 else 0.0

    return confmat, metric_logger.meters["loss"].global_avg, avg_dice


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | 🚀 Advanced Training Mode On")


    train_dataset = get_Dataset(args.data_path, train=True,
                                transforms=train_transform(size=args.train_size))
    val_dataset = get_Dataset(args.data_path, train=False,
                              transforms=eval_transform(size=args.train_size))

    num_workers = 2

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)


    model = create_model(args)
    model.to(device)


    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = GradScaler()


    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_loader),
        args.epochs,
        warmup=True
    )

    # 5. 损失函数
    criterion = CriterionComposite(
        num_classes=args.num_classes,
        alpha=0.5
    ).to(device)

    best_miou = 0.
    start_time = time.time()

    # 创建模型保存目录
    model_save_dir = os.path.join("save_weights", args.model)
    os.makedirs(model_save_dir, exist_ok=True)

    # 日志路径改到模型目录下
    log_path = os.path.join(model_save_dir, f"log_{args.model}.csv")

    base_columns = ['train_loss', 'val_loss', 'acc',
                    'miou', 'recall', 'precision',
                    'dice', 'f1']
    class_iou_columns = [f'class{i}_iou' for i in range(args.num_classes)]

    df = pd.DataFrame(columns=base_columns + class_iou_columns)

    acc_steps = 4

    for epoch in range(args.start_epoch, args.epochs):

        train_loss = train_one_epoch_pro(
            model, optimizer, train_loader, device, epoch,
            criterion=criterion,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            accumulation_steps=acc_steps
        )

        # 验证
        confmat, val_loss, dice = textevaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=args.num_classes,
            criterion=criterion
        )

        matrix = confmat.compute()

        def to_num(val):
            if torch.is_tensor(val):
                return val.item()
            return float(val)

        acc_val = to_num(matrix[0])
        miou_val = to_num(matrix[2])
        dice_val = to_num(dice)

        if torch.is_tensor(matrix[1]):
            iou_per_class = matrix[1].cpu().tolist()
        else:
            iou_per_class = list(matrix[1])

        def get_avg_metric(metric_data):
            if torch.is_tensor(metric_data):
                m = metric_data[1:] if len(metric_data) > 1 else metric_data
                return m.mean().item()
            else:
                m = metric_data[1:] if len(metric_data) > 1 else metric_data
                return sum(m) / len(m) if len(m) > 0 else 0

        recall_val = get_avg_metric(matrix[3])
        precision_val = get_avg_metric(matrix[4])
        f1_val = get_avg_metric(matrix[5])

        print(f"📊 Epoch {epoch} Result: "
              f"Avg mIoU = {miou_val * 100:.2f}%, "
              f"Dice = {dice_val:.4f}")

        row_data = [
                       train_loss,
                       val_loss,
                       acc_val,
                       miou_val * 100,
                       recall_val,
                       precision_val,
                       dice_val,
                       f1_val
                   ] + [float(iou) * 100 for iou in iou_per_class]

        df.loc[epoch] = row_data

        df.to_csv(log_path, index=False)

        # 保存最优模型
        if miou_val >= best_miou:
            best_miou = miou_val
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args
            }

            torch.save(save_file,
                       os.path.join(model_save_dir, "best_model.pth"))

            print(f"💾 Saved best model at epoch {epoch} "
                  f"(mIoU: {best_miou * 100:.2f}%)")

    total_time = time.time() - start_time
    print(f"✅ Total training time: "
          f"{str(datetime.timedelta(seconds=int(total_time)))}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced FMB Training")
    parser.add_argument("--model",
                        default="IRSegText-NUDT_SIRST_1_1_point_halfnotrainclip_noalinglossnoattentionloss")  # IRSTD_1K_point

    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--data-path", default=r"D:\zyl\IRSeg\irseg\data\dataset\NUDT_SIRST_1_1_point")
    parser.add_argument("--device", default="cuda:0")


    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--train_size", default=256, type=int)


    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-2, type=float)

    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    set_seed(2026)  # 你可以换成任意整数
    args = parse_args()
    if not os.path.exists(f"./save_weights/{args.model}"):
        os.makedirs(f"./save_weights/{args.model}")
    main(args)