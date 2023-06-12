import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
import matplotlib.pyplot as plt

from .ema import EMA
# from ema import EMA


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. 
    Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module):model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int):number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay # 指数移动平均衰减
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """

    def __init__(
            self,
            model,
            img_size,
            img_channels,
            num_classes,
            betas,
            loss_type="l2",
            use_ema = False,
            ema_decay=0.9999,
            ema_start=1,
            ema_update_rate=1,
            
    ):
        super().__init__()
        self.seed = 1

        self.model = model
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = deepcopy(model)
            self.ema = EMA(ema_decay)
            # EMA是指指数移动平均（Exponential Moving Average）的缩写。在这个代码中，EMA被用来对模型的权重进行平滑处理。
            # 99.9%旧模型参数+0.1%新模型参数
            self.ema_decay = ema_decay
            self.ema_start = ema_start
            self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)  # 累乘

        to_torch = partial(torch.tensor, dtype=torch.float32)  # partial函数的作用是将一个函数的一部分参数固定下来,返回一个新的函数。

        self.register_buffer("betas", to_torch(betas))  # buffer变量不需要进行梯度计算
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        # 到指定的更新频率再更新EMA模型
        self.step += 1
        if self.use_ema:
            if self.step % self.ema_update_rate == 0:
                if self.step < self.ema_start:
                    self.ema_model.load_state_dict(self.model.state_dict())
                else:
                    self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y):
        if self.use_ema:
            z = self.ema_model(x, t, y)
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * z) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )  
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
            
    @torch.no_grad()
    def sobel(self,img=torch.rand((1, 1, 100, 100))):
        # 定义Sobel算子的x方向和y方向kernel
        sobel_kernel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  
        sobel_kernel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        
        # 对图像数据求Sobel梯度
        grad_x = torch.nn.functional.conv2d(img, sobel_kernel_x.unsqueeze(0).unsqueeze(0), padding=1) 
        grad_y = torch.nn.functional.conv2d(img, sobel_kernel_y.unsqueeze(0).unsqueeze(0), padding=1)

        # 计算Sobel梯度模长
        grad = grad_x.pow(2) + grad_y.pow(2)
        grad = grad.sqrt()
        return grad
    
    
    @torch.no_grad()
    def sample(self, batch_size, device, y=None):
        """采样方法."""
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        torch.manual_seed(self.seed)
        self.seed += 1
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device) # 3没有添加edge

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y)

            if t > 0:
                torch.manual_seed(self.seed)
                self.seed += 1
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()
    

    
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None):
        """采样扩散序列"""
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        torch.manual_seed(self.seed)
        self.seed += 1
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        """
        对x进行扰动前向一步加噪过程
        self包含了一些参数，如betas,alphas等
        x: 原始图像
        t: 当前batch的时间步
        noise: 噪声
        """

        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t, y):
        # 获取损失值
        torch.manual_seed(self.seed)
        self.seed += 1
        
        noise = torch.randn_like(x)  # 生成一个与x相同size的服从标准正态分布的噪声
        # 256, 3, 32, 32
        perturbed_x = self.perturb_x(x, t, noise)  # 对x进行扰动
        
        # 256, 3, 32, 32 
        estimated_noise = self.model(perturbed_x, t, y)  # 估计噪声
        
        # self.sample(estimated_noise.shape[0],estimated_noise.device)
        # 256, 3, 32, 32
        
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None):
        b, c, h, w = x.shape
        # print("x.shape", x.shape)
        device = x.device
        if h != self.img_size[0] or h != w:
            raise ValueError("image height does not match diffusion parameters")
        torch.manual_seed(self.seed)
        self.seed += 1
        t = torch.randint(0, self.num_timesteps, (b,), device=device)  # 在[0, num_timesteps)中随机生成b个数
        
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008):
    print("generate_cosine_schedule")

    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    alphas = []
    f0 = f(0, T)
    t = [value for value in range(0, T + 1)]
    for i in t:
        alphas.append(f(i, T) / f0)

    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    return np.array(betas)


def generate_linear_schedule(T, low, high):
    print("generate_linear_schedule")
    betas = np.linspace(low * 1000 / T, high * 1000 / T, T)
    return betas


def generate_DNS_schedule(T, low, high, off=5):
    # Dynamic negative square.
    print("generate_DNS_schedule, T =",off)
    low = low * 1000 / T
    high = high * 1000 / T
    t = [value for value in range(0, T)]
    wt = []
    for i in t:
        w = -((i + off) / (T + 1 + off)) ** 2 + 1  # T+1防止最后失效
        wt.append(w)
    wt = np.array(wt)
    assert (high > low and low >= 0 and high <= 1), "high > low and low >= 0 and high <= 1"
    betas = (1 - wt) * (high - low) + low
    return betas


def extract(a, t, x_shape):
    """
    从a中抽取元素,并reshape成(b,1,1,1,1...) 1的个数等于len(x_shape)-1
    a : sqrt_alphas_cumprod 超参数
    t : 时间步 time_step
    x_shape : 输入图片x的shape

    Example:
        extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
    """

    b, *_ = t.shape  # *_忽略其他元素  b : 当前处理的图片数量
    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 第一个*是展开的操作


def get_alphas_cumprod(betas):
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)  # 累乘
    return alphas_cumprod


def test_schedule():
    """测试generate_schedule"""
    T = 2000
    betas_cos = generate_cosine_schedule(T)
    betas_lin = generate_linear_schedule(T, 0.0001, 0.02)
    betas_DNS5 = generate_DNS_schedule(T, 0.0001, 0.02, 5)
    betas_DNS500 = generate_DNS_schedule(T, 0.0001, 0.02, 500)
    betas_DNST = generate_DNS_schedule(T, 0.0001, 0.02, T)

    fig, ax = plt.subplots()
    ax.plot(np.arange(T), get_alphas_cumprod(betas_cos), label='Cosine', color='r')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_lin), label='Linear', color='g')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNS5), label='DNS5', color='c')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNS500), label='DNS500', color='y')
    ax.plot(np.arange(T), get_alphas_cumprod(betas_DNST), label='DNST', color='k')

    plt.xlabel('Iteration')
    plt.ylabel('Alpha_bar')
    plt.title('Decay Schedule')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("alpha_bar_schedule.png")


def test_extract():
    """测试提取函数"""
    hyperPara = torch.tensor([[1, 2, 3], [2, 3, 5], [3, 6, 8]])
    t = torch.tensor([[1], [2]])  # time_step
    x_shape = (2, 3, 4, 4)  # 两张4*4的三通道图片
    out = extract(hyperPara, t, x_shape)  # out是2,1,1,1的tensor
    print(out)


if __name__ == '__main__':
    test_schedule()
    pass
