"""
@author:  Zhedong Zheng
@contact: zdzheng12@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from market3d import Market3D
from model import Model, Model_dense, Model_dense2
from dgl.data.utils import download, get_download_dir
import torch.backends.cudnn as cudnn
from functools import partial
import tqdm
import urllib
import os
import argparse
import yaml
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from DGCNN import DGCNN
from pointnet2_model import PointNet2SSG, PointNet2MSG
from utils import CrossEntropyLabelSmooth

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--feature_dims',default='64,128,256,512', type=str,help='64, 128, 256, 512 or 64, 128, 256, 512, 1024')
parser.add_argument('--adam', action='store_true', help='use all training data' )
parser.add_argument('--dataset-path', type=str, default='./2DMarket/')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--name', type=str, default='basic')
parser.add_argument('--cluster', type=str, default='xyz')
parser.add_argument('--conv', type=str, default='EdgeConv', help='EdgeConv|GATConv')
parser.add_argument('--num-epochs', type=int, default=150)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--npart', type=int, default=1)
parser.add_argument('--init_points', type=int, default=512)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--lrRate', type=float, default=0.01)
parser.add_argument('--droprate', type=float, default=0.5)
parser.add_argument('--res_scale', type=float, default=1.0)
parser.add_argument('--resume', action='store_true', help='resume training' )
parser.add_argument('--flip', action='store_true', help='flip' )
parser.add_argument('--use_dense', action='store_true', help='use dense' )
parser.add_argument('--use_DGCNN', action='store_true', help='use DGCNN' )
parser.add_argument('--use_SSG', action='store_true', help='use SSG' )
parser.add_argument('--use_MSG', action='store_true', help='use MSG' )
parser.add_argument('--norm', action='store_true', help='use normalized data' )
parser.add_argument('--use_dense2', action='store_true', help='use dense' )
parser.add_argument('--id_skip', action='store_true', help='use dense' )
parser.add_argument('--scale', action='store_true', help='enable scale' )
parser.add_argument('--step', action='store_true', help='enable step lrRate' )
parser.add_argument('--rotate', action='store_true', help='enable rotate' )
parser.add_argument('--bg', action='store_true', help='enable background' )
parser.add_argument('--light', action='store_true', help='enable light model' )
parser.add_argument('--jitter', action='store_true', help='enable graph jitter' )
parser.add_argument('--D2', action='store_true', help='enable graph jitter' )
parser.add_argument('--no_se', action='store_true', help='enable graph jitter' )
parser.add_argument('--erase', type=float, default=0)
parser.add_argument('--no_xyz', action='store_true')
parser.add_argument('--train_all', action='store_true')
parser.add_argument('--slim', type=float, default=0.5)
parser.add_argument('--drop_connect_rate', type=float, default=0.0)
parser.add_argument('--channel', type=int, default=6)
parser.add_argument('--warm_epoch', default=-1, type=int, help='stride')
parser.add_argument('--labelsmooth', action='store_true', help='use label smooth' )
opt = parser.parse_args()

num_workers = opt.num_workers
batch_size = opt.batch_size

if not opt.resume:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids

if len(opt.feature_dims)>0:
    str_features = opt.feature_dims.split(',')
    features = []
    for feature in str_features:
        feature = int(feature)
        features.append(feature)
    opt.feature_dims = features
# set gpu ids
if len(opt.gpu_ids)>0:
    cudnn.enabled = True
    cudnn.benchmark = True

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train(model, optimizer, scheduler, train_loader, dev, epoch):

    model.train()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    if opt.labelsmooth:
        criterion = CrossEntropyLabelSmooth()
    warm_up = min(1.0,  0.1 + 0.9 * epoch / opt.warm_epoch) 

    warm_iteration = round(dataset_sizes['train']/opt.batch_size)*opt.warm_epoch # first 5 epoch
    total_iteration = round(dataset_sizes['train']/opt.batch_size)*opt.num_epochs

    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev).squeeze().long()
            optimizer.zero_grad()
            xyz = data[:,:,0:3].contiguous()
            rgb = data[:,:,3:].contiguous()
            logits = model(xyz.detach(), rgb.detach(), istrain=True)
            #loss = compute_loss(logits, label)
            if opt.npart>1: 
                loss = criterion(logits[0], label)
                for i in range(1, opt.npart): 
                    loss += criterion(logits[i], label)
            else:
                loss = criterion(logits, label)
            if epoch<opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            loss.backward()
            optimizer.step()

            if opt.npart>1:
                logit_sum = logits[0].detach()
                for i in range(1, opt.npart): 
                    logit_sum += logits[i].detach()
                _, preds = logit_sum.max(1)
            else:
                _, preds = logits.max(1)

            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss.item()
            total_correct += correct

            tq.set_postfix({
                #'Loss': '%.5f' % loss,
                'AvgLoss': '%.4f' % (total_loss / num_batches),
                #'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.4f' % (total_correct / count)})
        y_loss['train'].append(total_loss / num_batches)
        y_err['train'].append(1.0-total_correct / count)

    scheduler.step()

def evaluate(model, test_loader, dev):
    model.eval()

    total_correct = 0
    count = 0
    val_loss =0 
    val_loss_sum =0 
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq: # 6890,6
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                xyz = data[:,:,0:3].contiguous()
                rgb = data[:,:,3:].contiguous()
                logits = model(xyz.detach(), rgb.detach(), istrain=False)
                if opt.npart>1:
                    val_loss = criterion(logits[0], label)
                    val_logit_sum = logits[0]
                    for i in range(1, opt.npart):
                        val_loss += criterion(logits[i], label)
                        val_logit_sum += logits[i]
                else:
                    val_loss = criterion(logits, label)

                if opt.npart>1:
                    _, preds = val_logit_sum.max(1)
                else:
                    _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples
                val_loss_sum += val_loss
                num_batches += 1
                tq.set_postfix({
                    'AvgLoss': '%.4f' % (val_loss_sum / num_batches),
                    'AvgAcc': '%.4f' % (total_correct / count)})
            y_loss['val'].append( val_loss_sum/num_batches )
            y_err['val'].append(1.0-total_correct / count)    

    return total_correct / count

######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./snapshot',opt.name,'train.png'))


market_data = Market3D(opt.dataset_path, flip=opt.flip, slim=opt.slim, norm = opt.norm, scale = opt.scale, erase=opt.erase, rotate = opt.rotate, channel = opt.channel, bg=opt.bg, D2 = opt.D2)

train_loader = CustomDataLoader(market_data.train())
if opt.train_all:
    train_loader = CustomDataLoader(market_data.train_all())
valid_loader = CustomDataLoader(market_data.valid())

dataset_sizes = {}
dataset_sizes['train'] = market_data.train().img_num
dataset_sizes['val'] = market_data.valid().img_num

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.class_num = market_data.train().class_num

if opt.use_dense:
    model = Model_dense(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, use_se = not opt.no_se, graph_jitter = opt.jitter )
elif opt.use_dense2: 
    model = Model_dense2(opt.k, opt.feature_dims, [512],  output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate=opt.drop_connect_rate, res_scale =opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, graph_jitter = opt.jitter)
elif opt.use_DGCNN:
    model = DGCNN( 20, [64,128,256,512], [512,512], output_classes=opt.class_num,  input_dims=3, dropout_prob=opt.droprate )
elif opt.use_SSG:
    model = PointNet2SSG(output_classes=opt.class_num, init_points = 512, input_dims=3, dropout_prob=opt.droprate, use_xyz = not opt.no_xyz )
elif opt.use_MSG:
    model = PointNet2MSG(output_classes=opt.class_num, init_points = 512, input_dims=3, dropout_prob=opt.droprate, use_xyz = not opt.no_xyz )
else:
    model = Model(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate , npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, graph_jitter = opt.jitter)
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of training parameters: %.2f M'% (params/1e6) )
#model = model.to(dev)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
if opt.load_model_path:
    model.load_state_dict(torch.load(opt.load_model_path, map_location=dev))

optimizer = optim.SGD(model.parameters(), lr=opt.lrRate, momentum=0.9, nesterov=True, weight_decay=1e-4)
if opt.adam:
    optimizer = optim.Adam(model.parameters(), opt.lrRate, weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.num_epochs, eta_min=0.01*opt.lrRate)
if opt.step:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,140], gamma=0.1)

best_valid_acc = 0
best_test_acc = 0

if not os.path.exists('./snapshot/'):
    os.mkdir('./snapshot/')
save_model_path = './snapshot/' + opt.name
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
# save opts
with open('%s/opts.yaml'%save_model_path,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

for epoch in range(opt.num_epochs):
    print('Epoch #%d Validating' % epoch)
    if not opt.train_all:
        valid_acc = evaluate(model, valid_loader, dev)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_model_path + '/model_best.pth')
        print('Current validation acc: %.5f (best: %.5f)' % (
            valid_acc, best_valid_acc))
    torch.save(model.state_dict(), save_model_path + '/model_last.pth')
    train(model, optimizer, scheduler, train_loader, dev, epoch)
    if not opt.train_all:
        draw_curve(epoch)
