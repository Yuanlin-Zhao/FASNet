import torch
import time
from thop import profile, clever_format
from src.IRsegNetText import IRSegNet
# ===============================
# 你的模型
# ===============================
model = IRSegNet(
    in_channels=3,
    num_classes=2,
    base_c=32,
    deep_supervision=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ===============================
# 构造 Dummy 输入
# ===============================
batch_size = 1
H, W = 256, 256   # 你可以改成1024

dummy_img = torch.randn(batch_size, 3, H, W).to(device)

# CLIP text 输入长度 77 是标准
seq_len = 77
dummy_input_ids = torch.randint(0, 49408, (batch_size, seq_len)).to(device)
dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)

# ===============================
# 1️⃣ 计算 FLOPs & Params
# ===============================
with torch.no_grad():
    flops, params = profile(
        model,
        inputs=(dummy_img, dummy_input_ids, dummy_attention_mask),
        verbose=False
    )

flops, params = clever_format([flops, params], "%.3f")

print("========== Model Complexity ==========")
print("FLOPs:", flops)
print("Params:", params)

# ===============================
# 2️⃣ 计算 100 次平均推理时间
# ===============================
iterations = 100

# warmup
for _ in range(20):
    with torch.no_grad():
        _ = model(dummy_img, dummy_input_ids, dummy_attention_mask)

torch.cuda.synchronize()

start = time.time()

for _ in range(iterations):
    with torch.no_grad():
        _ = model(dummy_img, dummy_input_ids, dummy_attention_mask)

torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / iterations

print("========== Inference Time ==========")
print(f"Average inference time: {avg_time*1000:.3f} ms")
print(f"FPS: {1/avg_time:.2f}")