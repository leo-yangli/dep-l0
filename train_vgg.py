import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from datasets.dataloaders import cifar10, cifar100


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-p', type=int, default=300, help='print frequence')
parser.add_argument('--lr_vgg', type=float, default=0.05, help='initial learning rate for vgg model')
parser.add_argument('--lr_gate', type=float, default=0.001, help='initial learning rate for gates')
parser.add_argument('--lr_decay', type=float, default=0.2, help='learning rate decay')
parser.add_argument('---momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--fine_tune_epoch', type=int, default=0, help='fine tune epoch')
parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--lamba', type=float, default=1e-6, help='Coefficient for the L0 term')
parser.add_argument('--temp', type=float, default=2. / 3., help='temperature')
parser.add_argument('--epoch', type=int, default=300, help='epoch')
parser.add_argument('--save_epoch', type=int, default=20, help='save model epoch')
parser.add_argument('--net', type=str, default='dep', help='options: dep; hc')
parser.add_argument('--back_dep', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# print(args.gpu)
def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        if args.gpu is not None:
            images, labels = images.cuda(), labels.cuda()
        net.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels, net)
        loss.backward()
        optimizer_vgg.step()
        if optimizer_gate is not None:
            optimizer_gate.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        if n_iter % args.p == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer_vgg.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_loader.dataset)
            ))
            writer.add_scalar('Train/loss', loss.item(), n_iter)


def list2str(zs):
    zs_str = ''
    for z in zs:
        zs_str += str(z) + ', '
    return zs_str


def eval_training(epoch):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in val_loader:
        if args.gpu is not None:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels, net)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))

    writer.add_scalar('Test/Average loss', test_loss / len(val_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(val_loader.dataset), epoch)

    zs = []
    pis = []
    for k, layer in enumerate(net.layers):
        if hasattr(layer, 'qz_loga'):
            mode_z = layer.sample_z(1, sample=0).view(-1)
            writer.add_histogram('mode_z/{}'.format(layer.layer_name), mode_z.cpu().data.numpy(), epoch)
            writer.add_histogram('qz_loga/{}'.format(layer.layer_name), layer.qz_loga.cpu().data.numpy(), epoch)
            writer.add_scalar('z/{}'.format(layer.layer_name), (mode_z > 0).sum().cpu().data.numpy(), epoch)
            zs.append(int((mode_z > 0).sum().cpu().data.numpy()))
            pis.append(layer.qz_loga.cpu().data.numpy())
    writer.add_text('z', list2str(zs), epoch)
    return correct.float() / len(val_loader.dataset)


if __name__ == '__main__':
    # create model
    if args.net == 'dep':
        from models.vgg import vgg16_bn
    else:
        from models.vgg_hc import vgg16_bn

    net = vgg16_bn(args)
    if args.gpu is not None:
        net = net.cuda()

    # load dataset
    dataload = cifar10 if args.num_classes == 10 else cifar100
    train_loader, val_loader, num_classes = dataload(augment=True, batch_size=args.b)

    # loss function
    def loss_function(output, target_var, model):
        loss = nn.CrossEntropyLoss()(output, target_var)
        reg = model.regularization()
        total_loss = loss + reg
        return total_loss

    gate_params = []
    vgg_params = []
    for name, param in net.named_parameters():
        if 'loga' in name or 'gen' in name:
            gate_params.append(param)
        else:
            vgg_params.append(param)

    MILESTONES = [60, 120, 180, 240, 300]
    optimizer_vgg = optim.SGD(vgg_params, lr=args.lr_vgg, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_vgg = optim.lr_scheduler.MultiStepLR(optimizer_vgg, milestones=MILESTONES, gamma=args.lr_decay)

    optimizer_gate = optim.Adam(gate_params, lr=args.lr_gate)
    scheduler_gate = optim.lr_scheduler.MultiStepLR(optimizer_vgg, milestones=MILESTONES, gamma=args.lr_decay)

    CHECKPOINT_PATH = 'checkpoint'
    TIME_NOW = datetime.now().isoformat()
    checkpoint_path = os.path.join(CHECKPOINT_PATH, TIME_NOW)
    writer = SummaryWriter(
        log_dir='runs/{}_VGG16-{}-{}_CIFAR{}_{}'.format(TIME_NOW, args.net, 'backward' if args.back_dep else 'forward', args.num_classes, args.lamba))

    best_acc = 0.0
    # train
    for epoch in range(1, args.epoch):
        train(epoch)
        acc = eval_training(epoch)
        scheduler_vgg.step(epoch)
        scheduler_gate.step(epoch)
        if best_acc < acc:
            best_acc = acc
    # fine tune
    for epoch in range(args.epoch, args.epoch + args.fine_tune_epoch):
        net.fine_tune()
        train(epoch)
        acc = eval_training(epoch)
        if best_acc < acc:
            best_acc = acc

    print('best_acc:', best_acc)
    writer.close()
