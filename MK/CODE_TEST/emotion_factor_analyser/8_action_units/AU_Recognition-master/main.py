import argparse
import itertools

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

from au_data_loader import *
from helper import *
from Model import *

# <editor-fold desc="settings">
parser = argparse.ArgumentParser(description='AU Recognition')
parser.add_argument('data_path_dir', metavar='DATA_DIR', help='path to data dir')
parser.add_argument('label_path_dir', metavar='LABEL_DIR', help='path to label dir')
parser.add_argument('landmark_path_dir', metavar='LANDMARK_DIR', help='path to landmark dir')
parser.add_argument('--model', default='alexnet', metavar='MODEL',
                    help='alexnet or vgg16 or vgg16_bn or vgg19 or res18 or '
                         'res50 or res101 or inception (default: alexnet)')
# parser.add_argument('--emotion_path_dir', default=r'E:\DataSets\CKPlus\Emotion_labels\Emotion',
#                     metavar='DIR', help='path to emotion dir')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='numer of total epochs to run (default: 100)')
parser.add_argument('--step', default=20, type=int, metavar='N',
                    help='numer of epochs to adjust learning rate (default: 20)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--kfold', default=10, type=int, metavar='KFOLD',
                    help='kfold (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = np.inf


# </editor-fold>


def train(train_loader, model, criterion, optimizer, epoch, print_freq=5):
    losses = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)
        optimizer.zero_grad()

        # compute output
        output = model(input_var)
        loss0 = criterion(output[:, 0], target_var[:, 0])
        loss1 = criterion(output[:, 1], target_var[:, 1])
        loss2 = criterion(output[:, 2], target_var[:, 2])
        loss3 = criterion(output[:, 3], target_var[:, 3])
        loss4 = criterion(output[:, 4], target_var[:, 4])
        loss5 = criterion(output[:, 5], target_var[:, 5])
        loss6 = criterion(output[:, 6], target_var[:, 6])
        loss7 = criterion(output[:, 7], target_var[:, 7])
        loss8 = criterion(output[:, 8], target_var[:, 8])
        loss9 = criterion(output[:, 9], target_var[:, 9])
        loss10 = criterion(output[:, 10], target_var[:, 10])
        loss11 = criterion(output[:, 11], target_var[:, 11])
        loss12 = criterion(output[:, 12], target_var[:, 12])
        loss13 = criterion(output[:, 13], target_var[:, 13])
        loss14 = criterion(output[:, 14], target_var[:, 14])
        loss15 = criterion(output[:, 15], target_var[:, 15])
        loss16 = criterion(output[:, 16], target_var[:, 16])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16
        losses.update(loss.data[0], input.size(0))

        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i + 1, len(train_loader), loss=losses))


def valid(val_loader, model, criterion, print_freq=1):
    losses = AverageMeter()
    model.eval()
    return_pred, return_tar = [], []

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(async=True)
        # with torch.no_grad():
        input_var = Variable(input, volatile=True)
        target_var = Variable(target.cuda(async=True), volatile=True)
        # compute output
        output = model(input_var)
        loss0 = criterion(output[:, 0], target_var[:, 0])
        loss1 = criterion(output[:, 1], target_var[:, 1])
        loss2 = criterion(output[:, 2], target_var[:, 2])
        loss3 = criterion(output[:, 3], target_var[:, 3])
        loss4 = criterion(output[:, 4], target_var[:, 4])
        loss5 = criterion(output[:, 5], target_var[:, 5])
        loss6 = criterion(output[:, 6], target_var[:, 6])
        loss7 = criterion(output[:, 7], target_var[:, 7])
        loss8 = criterion(output[:, 8], target_var[:, 8])
        loss9 = criterion(output[:, 9], target_var[:, 9])
        loss10 = criterion(output[:, 10], target_var[:, 10])
        loss11 = criterion(output[:, 11], target_var[:, 11])
        loss12 = criterion(output[:, 12], target_var[:, 12])
        loss13 = criterion(output[:, 13], target_var[:, 13])
        loss14 = criterion(output[:, 14], target_var[:, 14])
        loss15 = criterion(output[:, 15], target_var[:, 15])
        loss16 = criterion(output[:, 16], target_var[:, 16])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16

        return_pred.extend(output.data.cpu().tolist())
        return_tar.extend(target.tolist())
        losses.update(loss.data[0], input.size(0))
        if (i + 1) % print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i + 1, len(val_loader), loss=losses))

    return return_tar, return_pred, np.mean(losses.avg)


def main():
    global best_prec, args
    args = parser.parse_args()
    data_path_dir = args.data_path_dir  # r'E:\DataSets\CKPlus\cohn-kanade-images'
    label_path_dir = args.label_path_dir  # r'E:\DataSets\CKPlus\FACS_labels\FACS'
    landmark_path_dir = args.landmark_path_dir  # r'E:\DataSets\CKPlus\Landmarks\Landmarks'
    # emotion_path_dir = args.emotion_path_dir  # r'E:\DataSets\CKPlus\Emotion_labels\Emotion'

    if args.model == 'inception':
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    reserved_set, reserved_label = get_reserved_set(label_path_dir)
    au_image = load_au_image_from_path(data_path_dir)
    au_label = load_au_label_from_path(label_path_dir, reserved_label, reserved_set)
    au_landmark = load_au_landmark_from_path(landmark_path_dir)
    for i in range(len(au_image)):
        au_image[i] = np.array(crop_au_img(au_image[i], au_landmark[i]))
    au_image = np.array(au_image)
    au_label = np.array(au_label)

    # build model
    model, criterion, optimizer = build_model()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    fold = args.kfold
    kf = KFold(fold, shuffle=True, random_state=20)
    res_tar, res_pred = [], []
    for k, (train_index, test_index) in enumerate(kf.split(au_image, au_label)):
        train_dataset = au_data_loader(au_image[train_index], au_label[train_index], transform=transform)
        valid_dataset = au_data_loader(au_image[test_index], au_label[test_index], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        if args.evaluate:
            tar, pred, _ = valid(valid_loader, model, criterion)
            res_pred.extend(pred)
            res_tar.extend(tar)
            continue

        # build a new model
        model, criterion, optimizer = build_model()

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr, args.step)
            train(train_loader, model, criterion, optimizer, epoch)
        tar, pred, ls = valid(valid_loader, model, criterion)
        res_pred.extend(pred)
        res_tar.extend(tar)

        # is_best = ls < best_prec
        # best_prec = min(ls, best_prec)
        # save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'best_prec': best_prec,
        #     'optimizer': optimizer.state_dict()
        # }, is_best, filename=args.model + '_model')
        print('fold: {0}\t loss: {1}'.format(k, ls))

    res_pred = np.array(res_pred)
    res_tar = np.array(res_tar)

    out = []
    threshold = 0.50
    mean = 0
    for i in range(res_tar.shape[1]):
        print()
        print('AU' + str(list(reserved_set)[i]) + ':' +
              str(round(f1_score(res_tar[:, i], (res_pred[:, i]>=threshold).astype(np.float32)), 4)))
        out.append('AU' + str(list(reserved_set)[i]) + ':' +
                   str(round(f1_score(res_tar[:, i], (res_pred[:, i]>=threshold).astype(np.float32)), 4)))
        mean += round(f1_score(res_tar[:, i], (res_pred[:, i]>=threshold).astype(np.float32)), 4)
        cm = confusion_matrix(res_tar[:, i], (res_pred[:, i]>=threshold).astype(np.float32))
        plt.figure()
        plot_confusion_matrix(cm, classes=[0, 1])
        # plt.figure()
        # plot_confusion_matrix(cm, classes=[0, 1], normalize=True)
        # plt.show()
        print()
    out.append("AU mean " + str(mean/17))

    # write to txt
    output_txt = str(args.model) + '_output.txt'
    with open(output_txt, 'w') as f:
        for i in out:
            f.writelines(i)
            f.write('\n')


def build_model(pretrained=True):
    if args.model == 'alexnet':
        model = alexnet(pretrained=pretrained)
    elif args.model == 'vgg16':
        model = vgg16(pretrained=pretrained)
    elif args.model == 'vgg16_bn':
        model = vgg16_bn(pretrained=pretrained)
    elif args.model == 'vgg19':
        model = vgg19(pretrained=pretrained)
    elif args.model == 'res18':
        model = resnet18(pretrained=pretrained)
    elif args.model == 'res50':
        model = resnet50(pretrained=pretrained)
    elif args.model == 'res101':
        model = resnet101(pretrained=pretrained)
    elif args.model == 'inception':
        model = inception_v3(pretrained=pretrained)
    else:
        raise ValueError('not supported model')
    model.cuda()
    criterion = nn.BCELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier1.parameters():
        param.requires_grad = True
    for param in model.classifier2.parameters():
        param.requires_grad = True
    for param in model.classifier3.parameters():
        param.requires_grad = True
    for param in model.classifier4.parameters():
        param.requires_grad = True
    for param in model.classifier5.parameters():
        param.requires_grad = True
    for param in model.classifier6.parameters():
        param.requires_grad = True
    for param in model.classifier7.parameters():
        param.requires_grad = True
    for param in model.classifier8.parameters():
        param.requires_grad = True
    for param in model.classifier9.parameters():
        param.requires_grad = True
    for param in model.classifier10.parameters():
        param.requires_grad = True
    for param in model.classifier11.parameters():
        param.requires_grad = True
    for param in model.classifier12.parameters():
        param.requires_grad = True
    for param in model.classifier13.parameters():
        param.requires_grad = True
    for param in model.classifier14.parameters():
        param.requires_grad = True
    for param in model.classifier15.parameters():
        param.requires_grad = True
    for param in model.classifier16.parameters():
        param.requires_grad = True
    for param in model.classifier17.parameters():
        param.requires_grad = True
    return model, criterion, optimizer


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()


# python main.py CKPlus/cohn-kanade-images CKPlus/FACS_labels/FACS CKPlus/Landmarks --model alexnet --epochs 100 -b 16 --step 10
