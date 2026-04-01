from collections import defaultdict, deque
import datetime
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import errno
import os
from torch import nn
from train_utils.dice_coefficient_loss import build_target, multiclass_dice_coeff


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        # h 是混淆矩阵 [num_classes, num_classes]
        h = self.mat.float()

        # 1. 计算全局准确率 (所有预测正确的像素 / 总像素)
        total_pixels = h.sum()
        acc_global = torch.diag(h).sum() / total_pixels if total_pixels > 0 else torch.tensor(0.0)

        # 转为 numpy 处理后续指标
        hist = h.cpu().numpy()
        diag = np.diag(hist)

        # 2. 计算 Recall 和 Precision
        # 使用 np.errstate 忽略除以0的警告，手动处理分母为0的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            # Recall = TP / (TP + FN)  -> hist.sum(1) 是每一类的 Ground Truth 总数
            recall = diag / hist.sum(1)
            # Precision = TP / (TP + FP) -> hist.sum(0) 是每一类预测的总数
            Precision = diag / hist.sum(0)

        # 3. 计算 IoU (核心改进：确保不出现 nan)
        # 分母 = TP + FP + FN
        iu_denominator = hist.sum(1) + hist.sum(0) - diag

        with np.errstate(divide='ignore', invalid='ignore'):
            iou_array = diag / iu_denominator

        # --- 关键步骤：只计算有值的类别 ---
        # 只有当该类在 GT 中出现过，或者被模型预测出来过（分母 > 0），该类才是有意义的
        present_mask = iu_denominator > 0
        if np.any(present_mask):
            # 仅对存在的类别取平均值
            miou = np.nanmean(iou_array[present_mask])
        else:
            miou = 0.0

        # 4. 计算 F1-score (基于 Precision 和 Recall 的调和平均)
        with np.errstate(divide='ignore', invalid='ignore'):
            F1 = 2 * (Precision * recall) / (Precision + recall)

        # 将所有的 nan 替换为 0.0，确保返回的数组干净
        recall = np.nan_to_num(recall)
        Precision = np.nan_to_num(Precision)
        iou_array = np.nan_to_num(iou_array)
        F1 = np.nan_to_num(F1)

        return acc_global.item(), iou_array, float(miou), recall, Precision, F1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        Accuracy,iou,miou,recall,Precision,f1= self.compute()
        return (
            '=========================================='
            '\nAccuracy: {:.3f}\n'
            'IoU: {}\n'
            'mean IoU: {:.3f}\n'
            'recall: {}\n'
            'Precision: {}\n'
            'F1:  {}\n'
        ).format(
                Accuracy * 100,
                ['{:.3f}'.format(i) for i in (iou * 100).tolist()],
                miou * 100,
                ['{:.3f}'.format(i) for i in (recall* 100).tolist()],
                ['{:.3f}'.format(i) for i in (Precision* 100).tolist()],
                ['{:.3f}'.format(i) for i in (f1* 100).tolist()])


class DiceCoefficient:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 记录全局的交集和总和
        self.inter = 0
        self.union = 0

    def update(self, output, target):
        # 1. 预处理：获取预测类别
        pred = output.argmax(1)

        # 2. 掩码处理：剔除 ignore_index
        keep = target != self.ignore_index

        # 3. 将 target 转为 one-hot 编码 (用于多类别计算)
        # 或者针对每一类单独计算
        for i in range(self.num_classes):
            p = (pred == i) & keep
            t = (target == i) & keep

            inter = (p & t).sum().float()
            union = p.sum().float() + t.sum().float()

            self.inter += inter
            self.union += union

    @property
    def value(self):
        # 计算全局 Dice
        # 增加极小值防止 nan
        return (2. * self.inter) / (self.union + 1e-7)

    def reduce_from_all_processes(self):
        # 如果使用分布式训练，需要同步 inter 和 union
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        torch.distributed.all_reduce(self.inter, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.union, torch.distributed.ReduceOp.SUM)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            pass
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'max memory: {memory:.0f} MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            yield obj
            if i % print_freq == 0:
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i,
                        len(iterable),
                        meters=str(self),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i,
                        len(iterable),
                        meters=str(self),
                    ))
            i += 1
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
