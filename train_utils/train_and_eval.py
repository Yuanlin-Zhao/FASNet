'''
import torch
from torch import nn
from train_utils import utils
from train_utils.dice_coefficient_loss import build_target,dice_loss
from train_utils.utils import DiceCoefficient


def criterion(inputs, target,loss_weight=None,ignore_index: int = -100):

    loss = nn.functional.cross_entropy(inputs, target, ignore_index = ignore_index, weight = loss_weight)
    dice_target = build_target(target, 2, ignore_index)
    loss  = loss +dice_loss(inputs, dice_target,ignore_index = 255)
    return loss


def evaluate(model,optimizer, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    dice = DiceCoefficient(num_classes = num_classes, ignore_index = 255)
    header = 'Val:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target,ignore_index=255)

            if isinstance(output,list):
                output = output[-1]
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
            dice.reduce_from_all_processes()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
        confmat.reduce_from_all_processes()

    return confmat,metric_logger.meters["loss"].global_avg,dice.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    lr_scheduler,print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)


    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target,ignore_index=255)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

'''


import torch
from torch import nn
import train_utils.utils as utils
from train_utils.dice_coefficient_loss import build_target, dice_loss


def criterion(inputs, target, num_classes, loss_weight=None, dice: bool = True, ignore_index: int = -100):
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss


def evaluate(model, optimizer, data_loader, device, num_classes):
    model.eval()

    confmat = utils.ConfusionMatrix(num_classes=num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Val:'
    total_dice = 0.0
    total_samples = 0

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device)
            target = target.to(device)

            output = model(image)

            loss = criterion(output, target, num_classes, loss_weight, ignore_index=255)

            # ---------- 计算 mIoU ----------
            pred_label = output.argmax(1)
            confmat.update(target.flatten(), pred_label.flatten())

            # ---------- 正确计算 Dice ----------
            if num_classes == 2:
                pred = torch.sigmoid(output[:, 1])
                pred = (pred > 0.5).float()
                target_bin = (target == 1).float()

                intersection = (pred * target_bin).sum(dim=(1, 2))
                union = pred.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2))

                dice = (2. * intersection + 1e-6) / (union + 1e-6)

                total_dice += dice.mean().item()
                total_samples += 1

            metric_logger.update(loss=loss.item())

    confmat.reduce_from_all_processes()

    avg_dice = total_dice / total_samples

    return confmat, metric_logger.meters["loss"].global_avg, avg_dice

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        print(f'当前类数为{num_classes}，未设置损失权重。')
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target,num_classes,loss_weight,ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
