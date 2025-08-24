# CIFAR-10 扩散概率模型实现

🇨🇳 中文版本 | [🇺🇸 English](README.md)

这是一个基于PyTorch的去噪扩散概率模型（DDPM）实现，专门用于在CIFAR-10数据集上生成高质量图像。本实现支持多种噪声调度策略、类条件生成以及各种架构改进。

## ✨ 特性

- **完整的DDPM实现**：完整实现去噪扩散概率模型
- **多种噪声调度**：支持线性、余弦和DNS（动态噪声调度）策略
- **UNet架构**：先进的UNet网络，包含注意力块和残差连接
- **类条件生成**：基于特定CIFAR-10类别的条件图像生成
- **EMA支持**：指数移动平均以提升样本质量
- **Wandb集成**：全面的实验跟踪和可视化
- **灵活训练**：可配置的超参数和早停机制
- **高质量采样**：支持多种配置的高效采样

## 🚀 快速开始

### 环境要求

```bash
pip install torch torchvision numpy matplotlib wandb
```

### 基本使用

1. **在CIFAR-10上训练DDPM模型：**
```bash
cd scripts
python train_cifar.py --iterations 800000 --batch_size 256 --log_to_wandb True
```

2. **从训练好的模型生成图像：**
```bash
python sample_images.py --model_path path/to/model.pth --save_dir ./generated --num_images 1000
```

## 📖 详细使用方法

### 训练配置

训练脚本支持丰富的配置选项：

```bash
python train_cifar.py \
    --learning_rate 2e-4 \
    --batch_size 256 \
    --iterations 800000 \
    --schedule DNS \
    --schedule_low 1e-4 \
    --schedule_high 0.02 \
    --use_ema True \
    --log_to_wandb True \
    --project_name "ddpm-cifar10"
```

#### 关键参数：

- `--schedule`：噪声调度类型（`l`为线性，`cos`为余弦，`DNS`为动态调度）
- `--use_labels`：启用类条件生成
- `--use_ema`：使用指数移动平均以获得更好的采样效果
- `--early_stop_loss_change`：训练稳定性的早停阈值
- `--base_channels`：UNet的基础通道数（默认：128）
- `--num_timesteps`：扩散时间步数（默认：1000）

### 采样选项

使用各种配置生成图像：

```bash
# 生成1000张无条件图像
python sample_images.py --model_path model.pth --save_dir ./output --num_images 1000

# 生成类条件图像（每类100张）
python sample_images.py --model_path model.pth --save_dir ./output --num_images 1000 --use_labels True
```

## 🏗️ 架构设计

### UNet主干网络
- **残差块**：具有时间和类别嵌入的深度残差连接
- **注意力机制**：在指定分辨率上的多头自注意力
- **灵活架构**：可配置的通道倍数和块深度
- **归一化**：可配置组数的组归一化

### 扩散过程
- **前向过程**：在T个时间步上逐渐添加高斯噪声
- **反向过程**：神经网络学习在每个时间步去噪
- **多种调度**：线性、余弦和DNS调度以适应不同的训练动态

### 训练特性
- **损失函数**：噪声预测的L1和L2损失选项
- **EMA模型**：用于稳定生成的指数移动平均
- **梯度裁剪**：自动梯度管理
- **Wandb日志**：实时训练指标和样本可视化

## 📊 噪声调度策略

本实现支持三种噪声调度策略：

1. **线性调度**：简单的噪声方差线性增加
2. **余弦调度**：基于余弦的方差平滑转换
3. **DNS（动态噪声调度）**：自适应调度以改善训练动态

## 🎯 实验结果

该模型在CIFAR-10上实现了高质量的图像生成：
- 清晰、详细的32x32 RGB图像
- 在所有10个类别上的多样化样本生成
- 具有早停机制的稳定训练
- 使用EMA模型的高效采样

## 📁 项目结构

```
ddpm_abarankab/
├── ddpm/                   # 核心实现
│   ├── __init__.py
│   ├── diffusion.py       # DDPM实现
│   ├── unet.py           # UNet架构
│   ├── script_utils.py   # 工具函数
│   ├── ema.py            # 指数移动平均
│   └── utils.py          # 通用工具
├── scripts/               # 训练和采样脚本
│   ├── train_cifar.py    # CIFAR-10训练脚本
│   └── sample_images.py  # 图像生成脚本
└── README.md             # 说明文档
```

## ⚙️ 配置选项

### 模型架构
- `base_channels`：基础通道数（默认：128）
- `channel_mults`：每个分辨率的通道倍数（默认：[1, 2, 2, 2]）
- `num_res_blocks`：每个分辨率的残差块数（默认：2）
- `attention_resolutions`：使用注意力的分辨率（默认：[1]）
- `dropout`：丢弃率（默认：0.01）

### 训练参数
- `learning_rate`：Adam优化器学习率（默认：2e-4）
- `batch_size`：训练批大小（默认：256）
- `iterations`：总训练迭代次数（默认：800000）
- `num_timesteps`：扩散时间步数（默认：1000）

### EMA设置
- `ema_decay`：EMA衰减率（默认：0.9999）
- `ema_update_rate`：更新频率（默认：1）
- `ema_start`：何时开始EMA（默认：1）

## 🔬 高级用法

### 自定义训练循环
```python
from ddpm import script_utils
import torch

# 使用自定义配置加载模型
args = create_argparser().parse_args()
diffusion = script_utils.get_diffusion_from_args(args)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

# 训练循环
for iteration, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = diffusion(x, y) if args.use_labels else diffusion(x)
    loss.backward()
    optimizer.step()
    diffusion.update_ema()
```

### 自定义采样
```python
# 生成指定数量的样本
samples = diffusion.sample(num_samples=64, device=device)

# 类条件采样
y = torch.arange(10, device=device).repeat(6, 1).flatten()[:64]
samples = diffusion.sample(64, device, y=y)
```

## 🤝 贡献指南

欢迎贡献！您可以：
- 报告错误和问题
- 建议新功能
- 提交拉取请求
- 改进文档

## 📄 许可证

本项目是开源的，遵循标准学术使用条款。

## 🙏 致谢

- Ho等人的原始DDPM论文
- PyTorch团队提供的优秀框架
- OpenAI的架构洞察
- 社区贡献和反馈

## 📞 联系方式

如有问题或合作需求：
- GitHub Issues：[在此报告](https://github.com/betterlmy/ddpm_abarankab/issues)
- Pull Requests：欢迎改进和建议

---

**祝您生成愉快！🎨**