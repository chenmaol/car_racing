import os
import argparse
import torch
import torch.distributed as dist
from datasets import get_loaders
from model import Model
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def evaluate(model, val_dataloader, val_sampler, rank, epoch):
    model.eval()
    val_sampler.set_epoch(epoch)  # 确保每个epoch中的数据不同

    key_num = len(args.keys)
    correct = np.zeros(key_num, dtype=np.int64)
    total = np.zeros(key_num, dtype=np.int64)

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            images, labels = data
            images, labels = images.to(rank), labels.to(rank)
            outputs = model(images)  # [b, t, 4, 2]
            for j in range(key_num):
                outputs_key = outputs[:, :, j, :].reshape(-1, outputs.shape[-1])
                labels_key = labels[:, :, j].view(-1)
                _, predicted_key = torch.max(outputs_key, 1)
                total_key = labels_key.size(0)
                correct_key = (predicted_key == labels_key).sum().item()
                correct[j] += correct_key
                total[j] += total_key

    global_correct = torch.IntTensor(correct).to("cuda").to(rank)
    global_total = torch.IntTensor(total).to("cuda").to(rank)
    dist.all_reduce(global_correct)
    dist.all_reduce(global_total)
    if rank == 0:
        print('=' * 20)
        print(f'evaluation epoch {epoch} accuracy:')
        print(f'overall: {100 * (global_correct / global_total).mean().item()}')
        for i in range(key_num):
            accuracy_key = 100 * global_correct[i] / global_total[i]
            print(f'    --key {i}: {accuracy_key.item():.4f}%')

    return (global_correct / global_total).mean().item()


def train_one_epoch(model, train_dataloader, train_sampler, rank, epoch, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    # 在每个epoch开始时设置epoch; make shuffling work properly across multiple epochs;
    # Otherwise, the same ordering will always be used.
    train_sampler.set_epoch(epoch)

    # 输出当前的学习率
    if rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} starting, current learning rate: {current_lr}")

    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(rank), labels.to(rank)
        optimizer.zero_grad()
        outputs = model(inputs)  # [b, t, 4, 2]
        outputs = outputs.view(-1, outputs.shape[-1])  # [b, t, 4, 2] -> [b*t*4, 2]
        labels = labels.view(-1)  # [b, t, 4]->[b*t*4, ], 0 or 1 binary labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += len(predicted)

        running_loss += loss.item() * len(predicted)
        if i % args.log_interval == 0 and rank == 0:
            print(f"[epoch: {epoch}, iteration: {i}] loss: {running_loss / running_total:.6f}, acc: {running_correct / running_total:.3f}")

    global_running_loss = torch.FloatTensor([running_loss]).to("cuda").to(rank)
    global_running_correct = torch.IntTensor([running_correct]).to("cuda").to(rank)
    global_running_total = torch.IntTensor([running_total]).to("cuda").to(rank)
    dist.all_reduce(global_running_loss)
    dist.all_reduce(global_running_correct)
    dist.all_reduce(global_running_total)
    return global_running_loss.item() / dist.get_world_size(), global_running_correct.item() / global_running_total.item()


def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    # torch2.0 Use torch run:
    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    # 准备数据 这里需要DistributedSampler
    train_dataloader, val_dataloader, train_sampler, val_sampler = get_loaders(args)

    # 初始化模型
    model = Model(args.model_config, args.batch_size).to("cuda").to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 每5个epoch将学习率减少为原来的0.1倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loss
    criterion = torch.nn.CrossEntropyLoss().to("cuda").to(local_rank)

    best_accuracy = 0.0  # 保存最佳准确率
    best_epoch = -1
    best_model_path = args.save_path

    if local_rank == 0:
        writer = SummaryWriter('runs/exp')
        argsDict = args.__dict__
        with open('args.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
    # 训练循环
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_dataloader, train_sampler, local_rank, epoch, optimizer, criterion)

        # 验证
        val_acc = evaluate(model, val_dataloader, val_sampler, local_rank, epoch)

        scheduler.step()  # 每个epoch结束时更新学习率

        # 只在主进程中保存模型
        if local_rank == 0 and val_acc > best_accuracy:
            best_epoch = epoch
            best_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with accuracy: {best_accuracy:.2f}%')

        if local_rank == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
    # # 加载最佳模型权重进行评估
    # if local_rank == 0:
    #     print(f"Loading best model from {best_model_path}")
    #     model.load_state_dict(torch.load(best_model_path))
    #     final_accuracy = evaluate(model, val_dataloader, val_sampler, local_rank, best_epoch)
    #     print(f"Final accuracy with the best model: {final_accuracy:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM training script")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--sequence_length', type=int, default=64, help='the sequence_length to be processed')
    parser.add_argument('--pred_seq_length', type=int, default=4, help='the sequence_length to be predict')
    parser.add_argument('--pred_gap_length', type=int, default=0, help='the time gap between frame seq and predict seq')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for optimizer')
    parser.add_argument('--data_dir', type=str, default="/raid/car_racing/FM/data", help='path to store data')
    parser.add_argument('--label_file', type=str, default="labels_interval-1_dirty-5.0.csv", help='path to store data')
    parser.add_argument('--data_stride', type=int, default=4, help='data pick interval')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker used in each thread to load data')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_path', type=str, default='./best_model.pth', help='path to save the best model')
    parser.add_argument('--keys', type=str, default=['w', 's', 'a', 'd'], nargs='+',help='keys predicted')
    parser.add_argument('--img_size', type=int, default=128, help='resize image')
    parser.add_argument('--model_config', type=str, default='model_config.yaml', help='model config file')

    args = parser.parse_args()
    main()

# 执行指令
# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
# --nproc_per_node：每个节点的进程数，通常设置为每台机器上的 GPU 数量。
# --nnodes：节点的数量，如果你只使用单台机器，这个值为 1。
# --node_rank：节点的排名，如果你只使用单台机器，这个值为 0。