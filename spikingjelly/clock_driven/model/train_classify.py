import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import os
from .. import functional

def cal_accuracy(output, target, topk=(1,)):
    # https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32).item()
            res.append(correct_k * (100.0 / batch_size))
        return res

def train_one_epoch(model, criterion, optimizer, data_loader, device, amp_scaler=None):
    model.train()
    train_acc1 = 0.
    train_acc5 = 0.
    train_loss = 0.
    samples_number = 0
    start_time = time.time()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()

        if amp_scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        functional.reset_net(model)

        acc1, acc5 = cal_accuracy(output, target, topk=(1, 5))
        train_acc1 += acc1
        train_acc5 += acc5
        train_loss += loss.item()
        samples_number += image.shape[0]

    train_acc1 /= samples_number
    train_acc5 /= samples_number
    train_loss /= samples_number

    print(f'train_acc1={train_acc1}, train_acc5={train_acc5}, train_loss={train_loss}, samples/s={samples_number / (time.time() - start_time)}')
    return train_acc1, train_acc5, train_loss

def evaluate(model, criterion, data_loader, device):
    model.train()
    test_acc1 = 0.
    test_acc5 = 0.
    test_loss = 0.
    samples_number = 0
    start_time = time.time()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target)

        functional.reset_net(model)

        acc1, acc5 = cal_accuracy(output, target, topk=(1, 5))
        test_acc1 += acc1
        test_acc5 += acc5
        test_loss += loss.item()
        samples_number += image.shape[0]

    test_acc1 /= samples_number
    test_acc5 /= samples_number
    test_loss /= samples_number

    print(f'test_acc1={test_acc1}, train_acc5={test_acc5}, train_loss={test_loss}, samples/s={samples_number / (time.time() - start_time)}')
    return test_acc1, test_acc5, test_loss

def train_eval_loop(args, model, criterion, optimizer, lr_scheduler, train_data_loader, test_data_loader, max_epoch, use_amp=False, tb_log_dir: str=None, pt_dir: str=None, resume_pt :str=None):

    start_epoch = 0
    if resume_pt is not None:
        checkpoint = torch.load(resume_pt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    if use_amp:
        amp_scaler = amp.GradScaler()
    else:
        amp_scaler = None

    if tb_log_dir is not None:
        tb_writer = SummaryWriter(tb_log_dir, purge_step=start_epoch)
    else:
        tb_writer = None

    max_test_acc1 = -1.
    test_acc5_at_max_test_acc1 = -1.
    max_test_acc5 = -1.


    for epoch in (start_epoch, max_epoch):
        print(f'epoch={epoch}, args={args}')
        loss, acc1, acc5 = train_one_epoch(model, criterion, optimizer, train_data_loader, args.device, amp_scaler)
        if tb_writer is not None:
            tb_writer.add_scalar('train_loss', loss, epoch)
            tb_writer.add_scalar('train_acc1', acc1, epoch)
            tb_writer.add_scalar('train_acc5', acc5, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss, acc1, acc5 = evaluate(model, criterion, test_data_loader, args.device)

        if tb_writer is not None:
            tb_writer.add_scalar('test_loss', loss, epoch)
            tb_writer.add_scalar('test_acc1', acc1, epoch)
            tb_writer.add_scalar('test_acc5', acc5, epoch)

        if pt_dir is not None:
            if lr_scheduler is not None:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                }
            else:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                }
            torch.save(checkpoint, os.path.join(pt_dir, 'ckp_latest.pt'))

        max_test_acc5 = max(acc5, max_test_acc5)
        if acc1 > max_test_acc1:
            max_test_acc1 = acc1
            test_acc5_at_max_test_acc1 = acc5
            if pt_dir is not None:
                torch.save(checkpoint, os.path.join(pt_dir, 'ckp_max_test_acc1.pt'))
        print(f'max_test_acc1={max_test_acc1}, max_test_acc5={max_test_acc5}, test_acc5_at_max_test_acc1={test_acc5_at_max_test_acc1}')






















