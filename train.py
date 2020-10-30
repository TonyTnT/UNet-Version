from datetime import datetime
import os
import random
import time
from datetime import datetime

from PIL import Image
from albumentations import (
    Resize, Normalize, Compose, HorizontalFlip, VerticalFlip, RandomCrop, ShiftScaleRotate,
    RandomBrightnessContrast,
)
from albumentations.pytorch import ToTensor
from apex import amp
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from data.data import ImgSeg
from loss import *
from models import *


def seg_transforms(phase, resize=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ Get segmentation albumentation tansforms
    Args:
        phase: train or valid
        resize: input image shape into model

    Returns:
        albu compose transforms
    Raises:
        IOError: An error occurred accessing ablumentation object.
    """
    assert (phase in ['train', 'valid', 'test'])
    transforms_list = []
    if phase == 'train':
        transforms_list.extend([
            # Rotate
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=20, border_mode=0, p=0.2),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            Resize(resize[0] + 64, resize[1] + 64, interpolation=Image.BILINEAR),
            Normalize(mean=mean, std=std, p=1),
            RandomCrop(resize[0], resize[1]),
            ToTensor(),
        ])
    else:
        transforms_list.extend([
            Resize(resize[0], resize[1], interpolation=Image.BILINEAR),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ])
    transforms = Compose(transforms_list)
    return transforms


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


def train(model, train_loader, optimizer, epoch, criterion, writer, use_apex=False, use_cuda=True, log_interval=10):
    # set model as training mode
    model.train()
    train_loss = 0
    dice_score = 0
    iou_score = 0
    N_count = len(train_loader.dataset)  # counting total trained sample in one epoch

    for batch_idx, (X, y) in enumerate(train_loader):
        X = Variable(X)
        y = Variable(y)
        # distribute data to device
        if use_cuda:
            X = X.cuda()
            y = y.cuda()
        # continue
        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)
        loss = criterion(output, y.long())
        train_loss += loss.item()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()  # loss要这么用
        else:
            loss.backward()
        optimizer.step()
        # to compute accuracy
        y_pred = torch.max(output, 1)[1]
        print(y_pred.max(), y_pred.min(), y.max(), y.min())
        # cal dice scores

        # dice_score += DiceLoss.BinaryDiceLoss(y_pred.data.cpu().long(), y.data.cpu().long())
        iou_score_iter = IOULoss.IOU_loss(y_pred, y).item()
        iou_score += iou_score_iter

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Dice: {:.6f}, IoU: {:.6f}'.format(
                epoch + 1, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(),
                dice_score, iou_score_iter))

    dice_score /= N_count
    iou_score /= N_count
    writer.add_scalar('epoch/Train_Loss', train_loss, epoch)
    writer.add_scalar('epoch/Train_Dice', dice_score, epoch)
    writer.add_scalar('epoch/Train_IoU', iou_score, epoch)
    writer.add_scalar('epoch/Train_LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Dice_Score: {:.2f}%, IoU_Score: {:.2f}%'.format(
        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
        dice_score, iou_score))
    return train_loss, iou_score


'''
def validation(model, test_loader, criterion, use_cuda=True):
    # set model as testing mode
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = Variable(X)
            y = Variable(y)
            # distribute data to device
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            output = model(X)

            loss = criterion(output, y)

            test_loss += loss.item()  # sum up batch loss

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            all_output.extend(output.cpu().tolist())
            all_w.extend(w.cpu().tolist())

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # save information
    writer.add_scalar('epoch/Val_Loss', val_loss, epoch)
    writer.add_scalar('epoch/Val_Acc', val_score, epoch)

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss,
                                                                                        100 * test_score))

    return test_loss, test_score, all_y.cpu().data.squeeze().tolist(), all_y_pred.cpu().data.squeeze().tolist()

'''
jpg_path = '/media/mdisk/chenx2/US/JPEGImages'
seg_path = '/media/mdisk/chenx2/US/Segmentation'
ratio_trainval = 0.3
batch_size = 16
num_workers = 4
model_name = 'UNet'
lossname = 'Dice'
learning_rate = 1e-3
epoch_num = 100
use_cuda = True
use_apex = False
ngpu = 2
img_size = 512
# set transformer
trans = seg_transforms(phase='train', resize=(img_size, img_size))

# random seed setting
seed = int(time.time())
# seed_torch(seed)

# data loading
dataset = ImgSeg(jpg_path, seg_path, transform=trans)
trainsize = int(len(dataset) * ratio_trainval)
testsize = len(dataset) - trainsize
train_set, val_set = torch.utils.data.random_split(dataset, [trainsize, testsize])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)

# model building
if model_name == 'UNet_3Plus_DeepSup_CGM':
    # UNet 3+ with deep supervision and class-guided module
    model = UNet_3Plus_DeepSup_CGM(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
elif model_name == 'UNet_3Plus_DeepSup':
    # UNet 3+ with deep supervision
    model = UNet_3Plus_DeepSup(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
elif model_name == 'UNet_3Plus':
    model = UNet_3Plus(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
elif model_name == 'UNet_2Plus':
    model = UNet_2Plus(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True)
elif model_name == 'UNet':
    # model = UNet(in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    model_name = Unet
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize all cnn parameters
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2 * epoch_num / 3, epoch_num], gamma=0.1)

if lossname == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif lossname == "CrossEntropy":
    criterion = nn.CrossEntropyLoss()
elif lossname == 'IOU':
    criterion = IOULoss.IOU()
elif lossname == "Dice":
    criterion = DiceLoss.DiceLoss()
elif lossname == "MSSSIM":
    criterion = MSSSIMLoss.MSSSIM()

if use_cuda:
    print('Using cuda')
    model = model.cuda()
    criterion = criterion.cuda()

if use_apex:
    print('Using apex')
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if ngpu > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAVEFOLD = '%s-%s' % (model_name, TIMESTAMP)

writer = SummaryWriter(logdir='logs/' + SAVEFOLD)
for epoch in range(epoch_num):
    torch.backends.cudnn.enabled = True
    train_loss, train_score = train(model, train_loader, optimizer, epoch, criterion, writer, use_apex=use_apex,
                                    use_cuda=use_cuda, log_interval=10)
    # adjust lr
    scheduler.step()
    # val_loss, val_score = validation()
    #
    # if val_score >= maxscore:
    #     maxscore = val_score
    #     # save Pytorch models of best record
    #     torch.save(model, os.path.join(SAVEFOLD, '%s_%s.pth' % (model_name, 'Best')))
