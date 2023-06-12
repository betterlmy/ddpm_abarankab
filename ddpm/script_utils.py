import argparse
import torchvision
import torch.nn.functional as F

from .unet import UNet
from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
    generate_DNS_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # RescaleChannels(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def str2bool(v):
    """
    将字符串的bool转换为Python的bool值
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)  # 添加参数


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        loss_type="l2",
        use_labels=False,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.01,
        activation="silu",
        attention_resolutions=(1,),
        use_ema=False,
        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    """
    从参数中获取扩散模型
    """
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=3,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 10,
        initial_pad=0,
    )

    if args.schedule == "cos":
        betas = generate_cosine_schedule(args.num_timesteps)

    elif args.schedule == "l":
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low,
            args.schedule_high,
        )
    elif args.schedule == "DNST":
        betas = generate_DNS_schedule(
            args.num_timesteps,
            args.schedule_low,
            args.schedule_high,
            off = args.num_timesteps
        )
    else:
        betas = generate_DNS_schedule(
            args.num_timesteps,
            args.schedule_low,
            args.schedule_high,
        )

    diffusion = GaussianDiffusion(
            model,
            (32, 32),
            3,
            10,
            betas,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            ema_update_rate=args.ema_update_rate,
            loss_type=args.loss_type,
        )

    return diffusion
