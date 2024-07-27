import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,4'
import argparse
import torch
import torch.distributed as dist
from datasets import get_loaders
from model import Model
import matplotlib.pyplot as plt
from net_args import network_args

def visualize_predictions(images, labels, predictions, epoch, local_rank):
    # 只在主进程中保存可视化结果
    if local_rank != 0:
        return

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    print('In visualize_predictions function, Images shape:', images.shape, 'Labels shape:', labels.shape,
          'Predictions shape:', predictions.shape)
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {labels[i]}\nPred: {predictions[i]}')

    plt.tight_layout()
    plt.savefig(f'visualization_epoch_{epoch}.png')
    plt.close()


def evaluate(model, val_dataloader, val_sampler, rank, epoch):
    model.eval()
    val_sampler.set_epoch(epoch)  # 确保每个epoch中的数据不同

    correct = {}
    total = {}

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            images, labels = data
            images, labels = images.to(rank), labels.to(rank)
            outputs = model(images)  # [b, t, 4, 2]
            key_num = outputs.shape[2]
            for j in range(key_num):
                outputs_key = outputs[:, :, j, :](-1, outputs.shape[-1])
                labels_key = labels[:, :, j].view(-1)
                _, predicted_key = torch.max(outputs_key, 1)
                total_key = labels_key.size(0)
                correct_key = (predicted_key == labels_key).sum().item()
                if j not in correct.keys():
                    correct[j] = correct_key
                    total[j] = total_key
                else:
                    correct[j] += correct_key
                    total[j] += total_key

            # 可视化部分MNIST结果
            # if i == 0:  # 只在第一个batch上进行可视化
            #     print('Images shape:', images.shape, 'Labels shape:', labels.shape, 'Predictions shape:',
            #           predicted.shape)
            #     visualize_predictions(images[:5], labels[:5], predicted[:5], epoch, rank)
    accuracy = 0
    for j in correct.keys():
        accuracy_key = 100 * correct[j] / total[j]
        accuracy += accuracy_key
        if rank == 0:
            print(f'epoch {epoch}: Accuracy of key {j} on the validation set: {accuracy_key:.2f}%')
    accuracy = accuracy / len(correct.keys())
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="IDM training script")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--sequence_length', type=int, default=32, help='the sequence_length to be processed')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--data_dir', type=str, default="/raid/car_racing/IDM/data", help='path to store data')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker used in each thread to load data')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_path', type=str, default='./best_model.pth', help='path to save the best model')
    args = parser.parse_args()


    

    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    # torch2.0 Use torch run:
    local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    # 准备数据 这里需要DistributedSampler
    train_dataloader, val_dataloader, train_sampler, val_sampler = get_loaders(args)

    # 初始化模型
    IDM_args = argparse.Namespace(**network_args)
    model = Model(IDM_args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 每5个epoch将学习率减少为原来的0.1倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # loss
    criterion = torch.nn.CrossEntropyLoss().to("cuda")

    best_accuracy = 0.0  # 保存最佳准确率
    best_epoch = -1
    best_model_path = args.save_path

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # 在每个epoch开始时设置epoch; make shuffling work properly across multiple epochs;
        # Otherwise, the same ordering will always be used.
        # train_sampler.set_epoch(epoch)

        # 输出当前的学习率
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} starting, current learning rate: {current_lr}")

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            labels = labels.type(torch.LongTensor) 
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)

            optimizer.zero_grad()
            outputs = model(inputs)  # [b, t, 4, 2]
            outputs = outputs.view(-1, outputs.shape[-1])  # [b, t, 4, 2] -> [b*t*4, 2]
            labels = labels.view(-1)  # [b, t, 4]->[b*t*4, ], 0 or 1 binary labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.log_interval == 0 and local_rank == 0:
                print(f"[epoch: {epoch}, iteration: {i}] average batch loss: {running_loss / args.log_interval:.3f}")
                running_loss = 0.0

        accuracy = evaluate(model, val_dataloader, val_sampler, local_rank, epoch)
        scheduler.step()  # 每个epoch结束时更新学习率

        # 只在主进程中保存模型
        if local_rank == 0 and accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with accuracy: {best_accuracy:.2f}%')

    # 加载最佳模型权重进行评估
    if local_rank == 0:
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        final_accuracy = evaluate(model, val_dataloader, val_sampler, local_rank, best_epoch)
        print(f"Final accuracy with the best model: {final_accuracy:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# 执行指令
# torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
# --nproc_per_node：每个节点的进程数，通常设置为每台机器上的 GPU 数量。
# --nnodes：节点的数量，如果你只使用单台机器，这个值为 1。
# --node_rank：节点的排名，如果你只使用单台机器，这个值为 0。
