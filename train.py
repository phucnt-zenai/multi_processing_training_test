# having: normal train, DP train, DDP train
import argparse
from data_mnist import get_mnist_datasets
from model_resnet import ResNetClassifier

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim


import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import time

### Calculate peak memory per GPU
def get_peak_memory_mb_dp():
    # cho DataParallel (single process) lấy max across devices
    if not torch.cuda.is_available():
        return 0.0
    peaks = []
    for i in range(torch.cuda.device_count()):
        peaks.append(torch.cuda.max_memory_allocated(i))
    print('2 GPU_RAM: ', peaks)
    return max(peaks) / (1024 ** 2) if peaks else 0.0

def get_peak_memory_mb_single(device):
    if not torch.cuda.is_available():
        return 0.0
    # device can be 'cuda' or 'cuda:0' etc.
    try:
        dev_idx = torch.device(device).index if isinstance(device, str) else device
    except Exception:
        dev_idx = torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)



def train(model, train_loader, valid_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    total_start = time.time()

    # reset peak memory stats for all devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # synchronize before measuring time if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start

        # Validation 
        val_start = time.time()
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        val_time = time.time() - val_start

        # peak memory (MB)
        if args.dp:
            peak_mem_mb = get_peak_memory_mb_dp()
        else:
            peak_mem_mb = get_peak_memory_mb_single(args.device) if torch.cuda.is_available() else 0.0

        print(f'\nEpoch {epoch} summary: train_time={epoch_time:.3f}s val_time={val_time:.3f}s peak_gpu_mem={peak_mem_mb:.1f}MB')
        print(f'Validation set: Accuracy: {correct}/{len(valid_loader.dataset)} '
              f'({100. * correct / len(valid_loader.dataset):.2f}%)\n')

        if correct / len(valid_loader.dataset) > best_acc:
            best_acc = correct / len(valid_loader.dataset)  

            if args.dp:
                torch.save(model.module.state_dict(), 'best_model_dp.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')

    total_time = time.time() - total_start
    print(f'Total training time: {total_time:.3f}s')

'''
def train(model, train_loader, valid_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Validation 
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        print(f'\nValidation set: Accuracy: {correct}/{len(valid_loader.dataset)} '
              f'({100. * correct / len(valid_loader.dataset):.0f}%)\n')
        if correct / len(valid_loader.dataset) > best_acc:
            best_acc = correct / len(valid_loader.dataset)  

            if args.dp:
                torch.save(model.module.state_dict(), 'best_model_dp.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
'''

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = ResNetClassifier(in_channels=args.in_channels).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset, valid_dataset = get_mnist_datasets(random_seed=args.seed)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    best_acc = 0.0
    total_start = time.time()

    # reset peak memory for this device
    torch.cuda.reset_peak_memory_stats(rank)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and rank == 0:
                print(f'Rank {rank}, Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_time = time.time() - epoch_start

        # Validation (only rank 0 prints/holds checkpoint)
        if rank == 0:
            val_start = time.time()
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            val_time = time.time() - val_start

            print(f'\nRank {rank} Epoch {epoch} summary: train_time={epoch_time:.3f}s val_time={val_time:.3f}s')

            if correct / len(valid_loader.dataset) > best_acc:
                best_acc = correct / len(valid_loader.dataset)  
                torch.save(model.state_dict(), 'best_model_ddp.pth')

        # get local peak memory (bytes) and reduce (max) to rank 0 to print
        local_peak_bytes = torch.tensor(torch.cuda.max_memory_allocated(rank), device=device, dtype=torch.float64)
        dist.reduce(local_peak_bytes, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            peak_mem_mb = (local_peak_bytes.item()) / (1024 ** 2)
            print(f'Peak GPU memory across ranks (MB): {peak_mem_mb:.1f}')

    total_time = time.time() - total_start
    if rank == 0:
        print(f'Total training time (DDP, rank 0): {total_time:.3f}s')

    cleanup_ddp()

'''
def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = ResNetClassifier(in_channels=args.in_channels).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset, valid_dataset = get_mnist_datasets(random_seed=args.seed)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and rank == 0:
                print(f'Rank {rank}, Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Validation 
        if rank == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
            print(f'\nValidation set: Accuracy: {correct}/{len(valid_loader.dataset)} '
                  f'({100. * correct / len(valid_loader.dataset):.0f}%)\n')
            
            if correct / len(valid_loader.dataset) > best_acc:
                best_acc = correct / len(valid_loader.dataset)  
                torch.save(model.state_dict(), 'best_model_ddp.pth')
    
    cleanup_ddp()
'''
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST ResNet Training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training (default: cuda)')
    parser.add_argument('--ddp', action='store_true', help='use Distributed Data Parallel training')
    parser.add_argument('--dp', action='store_true', help='use Data Parallel training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--in_channels', type=int, default=1, help='number of input channels (default: 1 for MNIST)')

    args = parser.parse_args()

    global_seed(args.seed)

    if args.dp:
        model = ResNetClassifier(in_channels=args.in_channels).to(args.device)
        model = nn.DataParallel(model)

        train_dataset, valid_dataset = get_mnist_datasets(random_seed=args.seed)
        
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator = g)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        train(model, train_loader, valid_loader, args)
    
    elif args.ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(train_ddp,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
        
    
    else:
        model = ResNetClassifier(in_channels=args.in_channels).to(args.device)

        train_dataset, valid_dataset = get_mnist_datasets(random_seed=args.seed)
        
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator = g)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        train(model, train_loader, valid_loader, args)
    