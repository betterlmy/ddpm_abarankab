import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image,ImageOps
import numpy as np
from torchvision.transforms import transforms



@torch.no_grad()
def sobel(img=torch.rand((1, 1, 100, 100))):
    
    # 定义Sobel算子的x方向和y方向kernel
    sobel_kernel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(img.device)
    sobel_kernel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).to(img.device)
    
    if img.shape[1] == 3:
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        grayscale = 0.299 * r + 0.587 * g + 0.114 * b
        grayscale = grayscale.unsqueeze(1) # [256, 1, 32, 32]
    # 对图像数据求Sobel梯度
    grad_x = torch.nn.functional.conv2d(grayscale, sobel_kernel_x, padding=1) 
    grad_y = torch.nn.functional.conv2d(grayscale, sobel_kernel_y, padding=1)

    # 计算Sobel梯度模长
    grad = grad_x.pow(2) + grad_y.pow(2)
    grad = grad.sqrt()
    return grad

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class TimeEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    
    Input:
        x:tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0  # dim must be even
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        """
        实现了对时间序列数据进行嵌入的操作，基于正弦和余弦函数的方法,帮助模型捕捉时间序列数据中的时间信息。
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 首先计算了一个用于嵌入的指数
        emb = torch.outer(x * self.scale, emb)  # 然后将时间序列 x 乘以一个标度因子 scale,与指数向量相乘得到一个嵌入矩阵。
        # outer用于计算两个向量的外积。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 最后，将这个嵌入矩阵划分成两个部分，分别应用正弦和余弦函数，并将它们拼接在一起返回。
        return emb


class Downsample(nn.Module):
    __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be ceven")

        return self.downsample(x)


class Upsample(nn.Module):
    __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    __doc__ = r"""
    Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)  # 1x1卷积 用于将输入的特征图转换为查询、键和值。
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """
        提取qkv
        计算注意力权重
        获得加权值
        下投影通道维度
        残差连接
        """
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.

    Input:
        x:tensor of shape (N, in_channels, H, W)
        time_emb:time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y:classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            dropout,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
    ):
        super().__init__()

        self.activation = activation  # relu

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # time and class conditioning
        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        # 时间是连续的，所以用线性层 类别是离散的，所以用embedding
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        # nn.Identity() is a no-op module that returns its input
        # 就是说如果in_channels == out_channels，那么就不需要residual_connection 直接返回x
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)

    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


class UNet(nn.Module):

    def __init__(
            self,
            img_channels,
            base_channels,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,
            time_emb_scale=1.0,
            num_classes=None,
            activation=F.relu,
            dropout=0.1,
            attention_resolutions=(),
            norm="gn",
            num_groups=32,
            initial_pad=0,
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        self.init_conv = nn.Conv2d(img_channels+1, base_channels, 3, padding=1)  # 3x3卷积用来初始化通道

        self.downs = nn.ModuleList()  # 下采样模块
        self.ups = nn.ModuleList()  # 上采样模块

        channels = [base_channels]
        now_channels = base_channels

        # UNet的下采样部分
        for i, mult in enumerate(channel_mults):
            # mult: 1, 2, 2, 2 通道数的倍数 下采样4次
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # UNet的中间部分
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])
        # UNet的上采样部分
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, y=None):
        ip = self.initial_pad
        
        # 加入边缘分支
        edge = sobel(x)
        x = torch.cat([x,edge],dim=1)
        
        if ip != 0:
            x = F.pad(x, (ip,) * 4)  # 四个方向都填充

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")

            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        x = self.init_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb, y)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x


def save_png(img_tensor,name="test.png"):
    img = img_tensor.squeeze(0).squeeze(0).detach().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(name)


if __name__ == '__main__':
    # img = Image.open("/root/lmy/data/LIDC/LIDC_0000.png")
    img = Image.open("/root/lmy/data/test_img/1.png")
    img = ImageOps.grayscale(img)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])
    
    img = transform(img)
    img = img.unsqueeze(0)
    grad = sobel(img)
    # 导出边缘图
    save_png(grad,"test2.png")
    
