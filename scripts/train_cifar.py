import argparse
import datetime
import torch
import wandb
# import time
from torch.utils.data import DataLoader
from torchvision import datasets
import sys
sys.path.append("../")
sys.path.append("./")
# sys.path.append("/root/lmy/ddpm_abarankab/")
from ddpm import script_utils,utils


def main():

    args = create_argparser().parse_args()
    device = args.device
    utils.printArgs(args)
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        # if args.optim_checkpoint is not None:
            # optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            # 记录到wandb上
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='betterlmy',
                config=vars(args),
                name=args.run_name+args.schedule,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        train_dataset = datasets.CIFAR10(
            root='./cifar_train',
            train=True,
            download=True,
            transform=script_utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root='./cifar_test',
            train=False,
            download=True,
            transform=script_utils.get_transform(),
        )

        def cycle(dl):
            """
            https://github.com/lucidrains/denoising-diffusion-pytorch/
            """
            while True:
                for data in dl:
                    yield data

        train_loader = cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

        acc_train_loss = 0
        acc_train_loss_list = []
        for iteration in range(1, args.iterations + 1):
            if args.test_mode:
                print("iteration:", iteration)
            diffusion.train()
            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                print("log!!iteration:", iteration)
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()

                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)

                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                # 提前结束,保存模型
                acc_train_loss_list.append(acc_train_loss)
                
                if len(acc_train_loss_list) > 10:
                    acc_train_loss_list.pop(0)
                    all = 0
                    for i in range(0,9):
                        all += abs(acc_train_loss_list[i]-acc_train_loss_list[i+1])
                    avg_change = all/10
                    print(avg_change)
                    
                    if avg_change < args.early_stop_loss_change:
                        print("Early stopping")
                        model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                        # optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                        torch.save(diffusion.state_dict(), model_filename)
                        # torch.save(optimizer.state_dict(), optim_filename)
                        break

                if args.log_to_wandb:
                    wandb.log({
                        "test_loss": test_loss,
                        "train_loss": acc_train_loss,
                        "samples": [wandb.Image(sample) for sample in samples],
                    })

                acc_train_loss = 0

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                # optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                # torch.save(optimizer.state_dict(), optim_filename)

        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=256,  # batch_size不影响训练速度
        iterations=800000,
        early_stop_loss_change =  3e-4,
        log_to_wandb=True,
        log_rate=100,
        checkpoint_rate=10000,
        # log_rate = 1,
        # checkpoint_rate = 2000,
        log_dir="./ddpm_logs",
        project_name="ddpm-cifar10",
        run_name=datetime.datetime.now().strftime("%m-%d-%H-%M-"),
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule="DNS",
        schedule_low=1e-4,
        schedule_high=0.02,
        device=device,
        test_mode = False,
    )
    defaults.update(script_utils.diffusion_defaults())  # dict

    parser = argparse.ArgumentParser()  # 创建一个解析对象
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
