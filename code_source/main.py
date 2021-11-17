from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import code.rot_inv_cnn.get_data as get_data
from code.rot_inv_cnn.pcam_dataset import PatchCamelyon
from torch.utils.data import DataLoader

from code.rot_inv_cnn.models import Baseline, PolarBaseline, BaselineSmall
import code.rot_inv_cnn.custom_vgg as vgg
import code.rot_inv_cnn.custom_resnet as resnet

from time import gmtime, strftime


def process_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='TYPE',
                        help='dataset type')
    parser.add_argument('--data-path', type=str, default=None, metavar='TYPE',
                        help='dataset type')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--model-type', type=str, default='vgg11', metavar='TYPE',
                        help="choices: 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', "
                             "'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'")
    parser.add_argument('--filter-width', type=int, default=3, metavar='N',
                        help='width of conv filters')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='')
    parser.add_argument('--baseline-small', action='store_true', default=False,
                        help='baseline but shrunk 3 times')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--lr-timestep', type=int, default=5, metavar='M',
                        help='Number of epochs before triggering learning rate decrease on plateau')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--patience', type=int, default=None, metavar='M',
                        help='Early stopping patience or None')
    parser.add_argument('--reg', nargs='+', type=float, default=[1e-6], metavar='FACTOR',
                        help='Weight decay value')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--duplicate-train', action='store_true', default=False,
                        help='adds 90 rotations to training set')

    args = parser.parse_args()
    return args


def get_model(args, data_width, n_input_channels, device):
    n_filters = 60
    polar = False
    if args.baseline_small:
        n_filters = 20
    elif args.baseline:
        pass
    else:
        polar = True
    filter_width = args.filter_width

    if 'vgg' in args.model_type:
        model = getattr(vgg, args.model_type.lower())(conv_type='classical' if not polar else 'polar',
                                                      kernel_size=filter_width,
                                                      num_classes=1)
    elif 'resnet' in args.model_type:
        model = getattr(resnet, args.model_type.lower())(conv_type='classical' if not polar else 'polar',
                                                         kernel_size=filter_width,
                                                         num_classes=1)
    else:
        model = Baseline(data_width, n_filters, polar, n_input_channels)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    return model.float()


def early_stopping(val_losses, patience, monitor='loss'):
    # Do not stop until enough epochs have been made
    if len(val_losses) < patience:
        return False, None

    if monitor == 'loss':
        best_val_loss = np.min(val_losses)
        if not np.any(val_losses[-patience:] <= best_val_loss):
            return True, best_val_loss
        return False, None
    else:
        raise ValueError('monitor ROI %s is not supported' % monitor)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(args, model, loss_fct, train_loader, optimizer, epoch, writer):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fct(output, target)
        loss.backward()
        optimizer.step()
        correct = ((output > .5).float() == target).sum().item()
        losses.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} ({:.4f})    acc: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), np.mean(losses), loss.item(),
                       100. * correct / len(target)))

        writer.add_scalar('loss/training', loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('accuracy/training', 100. * correct / len(target), epoch * len(train_loader) + batch_idx)
        if epoch == 1 and batch_idx == 0:
            writer.add_graph(model, data)

    return np.mean(losses)


def test(args, model, loss_fct, test_loader, epoch, writer, best_loss_sofar, epoch_size, train_loss=None, is_val=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_fct(output, target).item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += ((output > .5).float() == target).sum().item()

    # test_loss /= len(test_loader.dataset)

    # print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     'Val' if is_val else 'Test',
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    test_loss /= len(test_loader)
    print('\n{} set: Average loss: {:.4f}    acc: {:.3f}\n'.format('Val' if is_val else 'Test', test_loss,
                                                                   100. * correct / len(test_loader.dataset)))

    writer.add_scalar('loss/' + 'val' if is_val else 'test', test_loss, epoch * epoch_size)
    if train_loss is not None:
        writer.add_scalar('loss/difference', test_loss - train_loss, epoch * epoch_size)
    writer.add_scalar('accuracy/' + 'val' if is_val else 'test', 100. * correct / len(test_loader.dataset),
                      epoch * epoch_size)

    if args.save_model and test_loss < best_loss_sofar:
        save_path = os.path.join(writer.log_dir, 'model_epoch{:3d}_{:.4f}_acc{:.2f}.pt'.format(
            epoch, test_loss, 100. * correct / len(test_loader.dataset)))
        torch.save(model.state_dict(), save_path)
    else:
        save_path = None

    return test_loss, save_path


def main(args, reg=None):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset != 'pcam':
        train_loader, val_loader, test_loader, *_, n_input_channels, data_width = \
            get_data.main(use_cuda, args.batch_size,
                          args.test_batch_size,
                          duplicate_train=args.duplicate_train,
                          duplicate_test=True,
                          dataset_type=args.dataset)
        loss_fct = F.nll_loss
    else:
        dataset_train = PatchCamelyon(args.data_path, mode='train', augment=True)
        dataset_valid = PatchCamelyon(args.data_path, mode='valid', )
        dataset_test = PatchCamelyon(args.data_path, mode='test', )
        train_loader = DataLoader(dataset_train, args.batch_size, True, num_workers=5)
        val_loader = DataLoader(dataset_valid, args.batch_size, False, num_workers=5)
        test_loader = DataLoader(dataset_test, args.batch_size, False, num_workers=5)
        data_width = 96
        n_input_channels = 3
        loss_fct = torch.nn.BCEWithLogitsLoss()

    model = get_model(args, data_width, n_input_channels, device)
    print(model)
    print('# parameters', sum(p.numel() for p in model.parameters()))

    writer_folder = os.path.join('runs', args.dataset)
    run_filename = '_'.join(list(map(str, [strftime("%Y-%m-%d_%H:%M:%S", gmtime()),
                                           args.model_type, args.baseline, args.filter_width,
                                           args.batch_size, args.lr, args.reg])))
    run_filepath = os.path.join(writer_folder, run_filename)
    writer = SummaryWriter(run_filepath)
    print('Writer file', os.path.abspath(run_filepath))
    if not os.path.exists(run_filepath):
        os.makedirs(run_filepath)
    with open(os.path.join(run_filepath, 'config'), 'w') as f:
        f.write(str(args))

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.reg[0] if reg is None else reg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg[0] if reg is None else reg)

    best_loss_sofar = 100.
    best_saved_model_sofar = None
    val_losses = []
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = ReduceLROnPlateau(optimizer, 'min', args.gamma, args.lr_timestep, True, threshold=1e-3)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, loss_fct, train_loader, optimizer, epoch, writer)
        val_loss, save_path = test(args, model, loss_fct, val_loader, epoch, writer, best_loss_sofar,
                                   len(train_loader), train_loss, is_val=True)
        if val_loss < best_loss_sofar:
            best_loss_sofar = val_loss
            best_saved_model_sofar = save_path
        val_losses.append(val_loss)

        writer.add_scalar('lr', get_lr(optimizer), epoch * len(train_loader))
        scheduler.step(val_loss, epoch)

        if args.patience is not None:
            do_stop, min_loss = early_stopping(val_losses, args.patience)
            if do_stop:
                print('Early stopping triggered after %d epochs with min loss' % args.patience, min_loss)
                if best_saved_model_sofar is not None:
                    model.load_state_dict(torch.load(best_saved_model_sofar))
                break
    test(args, model, loss_fct, test_loader, -1, writer, 10000., len(train_loader))

    if args.save_model:
        torch.save(model.state_dict(), "model_test.pt")


if __name__ == '__main__':
    import sys
    # Training settings
    args = process_args()
    print(args)
    if len(args.reg) > 1:
        import subprocess
        from multiprocessing import Process
        not_reg_argv = []
        for a in sys.argv[1:]:
            not_reg_argv.append(a)
            if a == '--reg':
                break
        original_cli = 'python -m code.rot_inv_cnn.main ' + ' '.join(not_reg_argv)

        def make_cli(reg_value):
            reg_cli = original_cli + ' ' + str(reg_value)
            subprocess.check_output(reg_cli.split(' '), env=dict(os.environ))

        pool = []
        for reg in args.reg:
            p = Process(target=make_cli, args=(reg,))
            p.start()
            pool.append(p)

        for p in pool:
            p.join()

        # import torch.multiprocessing as mp
        # mp.set_start_method('spawn', force=True)
        # from copy import deepcopy
        # processes = []
        #     p = mp.Process(target=main, args=(cpy_args, reg))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
    else:
        main(args)
