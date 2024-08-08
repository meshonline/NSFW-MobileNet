import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from torchvision.models import mobilenet_v3_small

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13486"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size):
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")

    ddp_setup(rank, world_size)

    NUM_EPOCHS = 50
    RESUME = False

    data_dir = './data'
    checkpoint_dir = './checkpoint'

    traindir = os.path.join(data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_set = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,]))

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=False, sampler=DistributedSampler(train_set), pin_memory=True, drop_last=True)

    classes = train_loader.dataset.classes
    if rank == 0:
        print(classes)

    model = mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(classes))

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    decayRate = 0.915
    scheduler = ExponentialLR(optimizer, gamma=decayRate)

    if RESUME:
        map_location = {'cuda:0': 'cuda:{}'.format(rank)}
        checkpoint = torch.load(f'{checkpoint_dir}/model.pth', map_location=map_location)
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        scheduler.load_state_dict(checkpoint['sch'])

    if torch.cuda.is_available() and world_size > 1:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    try:
        for epoch in range(0 if not RESUME else step+1, NUM_EPOCHS, 1):
            train_loader.sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data = Variable(data)
                target = Variable(target)
                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss
                mean_loss = total_loss / (i + 1)
                percent = (i + 1) * 100 // len(train_loader)
                if rank == 0:
                    print('\r[Epoch {:03d}/{:03d}] [Batch {:03d}/{:03d}] lr: {:.2e} Loss: {:.4f} {} {}%'.format(epoch+1, NUM_EPOCHS, i+1, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], mean_loss.item(), '▇'*(percent//10), percent), end='')
            scheduler.step()
            if rank == 0:
                print()
                checkpoint = {}
                checkpoint['step'] = epoch
                checkpoint['model'] = model.module.state_dict()
                checkpoint['opt'] = optimizer.state_dict()
                checkpoint['sch'] = scheduler.state_dict()
                torch.save(checkpoint, f'{checkpoint_dir}/model.pth')
                torch.save(model.module, f'{checkpoint_dir}/model_full.pth')
    except KeyboardInterrupt:
        if rank == 0:
            print()
        pass

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
