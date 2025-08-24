# DDPM Implementation for CIFAR-10

[🇨🇳 中文版本](README_zh.md) | 🇺🇸 English

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for high-quality image generation on CIFAR-10 dataset. This implementation features multiple noise schedules, class-conditional generation, and various architectural improvements.

## ✨ Features

- **Complete DDPM Implementation**: Full implementation of the denoising diffusion probabilistic model
- **Multiple Noise Schedules**: Support for linear, cosine, and DNS (Dynamic Noise Schedule) 
- **UNet Architecture**: Advanced UNet with attention blocks and residual connections
- **Class-Conditional Generation**: Generate images conditioned on specific CIFAR-10 classes
- **EMA Support**: Exponential Moving Average for improved sample quality
- **Wandb Integration**: Comprehensive experiment tracking and visualization
- **Flexible Training**: Configurable hyperparameters and early stopping
- **High-Quality Sampling**: Efficient sampling with various configurations

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib wandb
```

### Basic Usage

1. **Train a DDPM model on CIFAR-10:**
```bash
cd scripts
python train_cifar.py --iterations 800000 --batch_size 256 --log_to_wandb True
```

2. **Generate images from a trained model:**
```bash
python sample_images.py --model_path path/to/model.pth --save_dir ./generated --num_images 1000
```

## 📖 Detailed Usage

### Training Configuration

The training script supports extensive configuration options:

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

#### Key Parameters:

- `--schedule`: Noise schedule type (`l` for linear, `cos` for cosine, `DNS` for dynamic)
- `--use_labels`: Enable class-conditional generation
- `--use_ema`: Use Exponential Moving Average for better sampling
- `--early_stop_loss_change`: Early stopping threshold for training stability
- `--base_channels`: Base number of channels in UNet (default: 128)
- `--num_timesteps`: Number of diffusion timesteps (default: 1000)

### Sampling Options

Generate images with various configurations:

```bash
# Generate 1000 unconditional images
python sample_images.py --model_path model.pth --save_dir ./output --num_images 1000

# Generate class-conditional images (100 per class)
python sample_images.py --model_path model.pth --save_dir ./output --num_images 1000 --use_labels True
```

## 🏗️ Architecture

### UNet Backbone
- **Residual Blocks**: Deep residual connections with time and class embeddings
- **Attention Mechanisms**: Multi-head self-attention at specified resolutions
- **Flexible Architecture**: Configurable channel multipliers and block depths
- **Normalization**: Group normalization with configurable groups

### Diffusion Process
- **Forward Process**: Gradually adds Gaussian noise over T timesteps
- **Reverse Process**: Neural network learns to denoise at each timestep
- **Multiple Schedules**: Linear, cosine, and DNS schedules for different training dynamics

### Training Features
- **Loss Functions**: L1 and L2 loss options for noise prediction
- **EMA Models**: Exponential moving averages for stable generation
- **Gradient Clipping**: Automatic gradient management
- **Wandb Logging**: Real-time training metrics and sample visualization

## 📊 Noise Schedules

This implementation supports three noise scheduling strategies:

1. **Linear Schedule**: Simple linear increase in noise variance
2. **Cosine Schedule**: Smoother transitions with cosine-based variance
3. **DNS (Dynamic Noise Schedule)**: Adaptive scheduling for improved training dynamics

## 🎯 Results

The model achieves high-quality image generation on CIFAR-10:
- Sharp, detailed 32x32 RGB images
- Diverse sample generation across all 10 classes
- Stable training with early stopping mechanisms
- Efficient sampling with EMA models

## 📁 Project Structure

```
ddpm_abarankab/
├── ddpm/                   # Core implementation
│   ├── __init__.py
│   ├── diffusion.py       # DDPM implementation
│   ├── unet.py           # UNet architecture
│   ├── script_utils.py   # Utility functions
│   ├── ema.py            # Exponential Moving Average
│   └── utils.py          # General utilities
├── scripts/               # Training and sampling scripts
│   ├── train_cifar.py    # CIFAR-10 training script
│   └── sample_images.py  # Image generation script
└── README.md             # This file
```

## ⚙️ Configuration Options

### Model Architecture
- `base_channels`: Base channel count (default: 128)
- `channel_mults`: Channel multipliers per resolution (default: [1, 2, 2, 2])
- `num_res_blocks`: Residual blocks per resolution (default: 2)
- `attention_resolutions`: Resolutions with attention (default: [1])
- `dropout`: Dropout rate (default: 0.01)

### Training Parameters
- `learning_rate`: Adam optimizer learning rate (default: 2e-4)
- `batch_size`: Training batch size (default: 256)
- `iterations`: Total training iterations (default: 800000)
- `num_timesteps`: Diffusion timesteps (default: 1000)

### EMA Settings
- `ema_decay`: EMA decay rate (default: 0.9999)
- `ema_update_rate`: Update frequency (default: 1)
- `ema_start`: When to start EMA (default: 1)

## 🔬 Advanced Usage

### Custom Training Loop
```python
from ddpm import script_utils
import torch

# Load model with custom configuration
args = create_argparser().parse_args()
diffusion = script_utils.get_diffusion_from_args(args)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)

# Training loop
for iteration, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = diffusion(x, y) if args.use_labels else diffusion(x)
    loss.backward()
    optimizer.step()
    diffusion.update_ema()
```

### Custom Sampling
```python
# Generate specific number of samples
samples = diffusion.sample(num_samples=64, device=device)

# Class-conditional sampling
y = torch.arange(10, device=device).repeat(6, 1).flatten()[:64]
samples = diffusion.sample(64, device, y=y)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under standard academic use terms.

## 🙏 Acknowledgments

- Original DDPM paper by Ho et al.
- PyTorch team for the excellent framework
- OpenAI for architectural insights
- Community contributions and feedback

## 📞 Contact

For questions or collaboration:
- GitHub Issues: [Report here](https://github.com/betterlmy/ddpm_abarankab/issues)
- Pull Requests: Welcome improvements and suggestions

---

**Happy Generating! 🎨**
