import os
import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

def main():
    args = cfg.parse_args()

    # 自动检测可用的 GPU 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. **初始化网络**
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)

    # **使用多个 GPU**
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        net = torch.nn.DataParallel(net)  # 让 PyTorch 自动并行计算

    net.to(device)  # **将模型放到 GPU**
    net.to(dtype=torch.bfloat16)  # **转换精度**

    # **加载预训练模型**
    if args.pretrain:
        print(f"Loading pretrained model from {args.pretrain}")
        weights = torch.load(args.pretrain, map_location=device)
        net.load_state_dict(weights, strict=False)

    # **指定不同的优化层**
    sam_layers = list(net.module.sam_mask_decoder.parameters()) if hasattr(net, 'module') else list(net.sam_mask_decoder.parameters())
    mem_layers = (
        list(net.module.obj_ptr_proj.parameters()) +
        list(net.module.memory_encoder.parameters()) +
        list(net.module.memory_attention.parameters()) +
        list(net.module.mask_downsample.parameters())
        if hasattr(net, 'module') else
        list(net.obj_ptr_proj.parameters()) +
        list(net.memory_encoder.parameters()) +
        list(net.memory_attention.parameters()) +
        list(net.mask_downsample.parameters())
    )

    # **优化器**
    optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08) if sam_layers else None
    optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08) if mem_layers else None

    # **学习率调度**
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # **开启 TensorFloat-32 加速**
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # **日志、数据加载**
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''检查点路径 & TensorBoard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    os.makedirs(checkpoint_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    '''开始训练'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):
        torch.cuda.empty_cache()  # **释放显存**

        net.train()
        time_start = time.time()
        loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
        logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
        time_end = time.time()
        print(f'Time for training epoch {epoch}: {time_end - time_start:.2f} seconds')

        # **评估模型**
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            with torch.no_grad():
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)

            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))

        torch.cuda.empty_cache()  # **释放显存**

    writer.close()


if __name__ == '__main__':
    main()
