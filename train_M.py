import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from market3d import Market3D
from model import Model, Model_dense, Model_dense2
from model_efficient import ModelE_dense
from model_efficient2 import ModelE_dense2
from dgl.data.utils import download, get_download_dir
import torch.backends.cudnn as cudnn
from functools import partial
import tqdm
import urllib
import os
import argparse
import yaml
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from DGCNN import DGCNN
import swa_utils
from lookahead import Lookahead
from ptflops import get_model_complexity_info
from pointnet2_model import PointNet2SSG, PointNet2MSG
from utils import SAM, CrossEntropyLabelSmooth, L2norm, make_weights_for_balanced_classes, disable_bn, enable_bn
from circle_loss import CircleLoss, convert_label_to_similarity

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning
    
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--feature_dims',default='64,128,256,512', type=str,help='64, 128, 256, 512 or 64, 128, 256, 512, 1024')
parser.add_argument('--adam', action='store_true', help='use adam' )
parser.add_argument('--adamW', action='store_true', help='use adamW' )
parser.add_argument('--restart', action='store_true', help='use adamW' )
parser.add_argument('--dataset-path', type=str, default='./2DMarket/')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--name', type=str, default='basic')
parser.add_argument('--pretrained_name', type=str, default='None')
parser.add_argument('--cluster', type=str, default='xyz')
parser.add_argument('--conv', type=str, default='EdgeConv', help='EdgeConv|GATConv')
parser.add_argument('--num-epochs', type=int, default=150)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--npart', type=int, default=1)
parser.add_argument('--init_points', type=int, default=512)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--lrRate', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=32, help='gamma for circle loss')
parser.add_argument('--droprate', type=float, default=0.5)
parser.add_argument('--res_scale', type=float, default=1.0)
parser.add_argument('--resume', action='store_true', help='resume training' )
parser.add_argument('--flip', action='store_true', help='flip' )
parser.add_argument('--gem', action='store_true', help='use GeM' )
parser.add_argument('--use_dense', action='store_true', help='use dense' )
parser.add_argument('--use_DGCNN', action='store_true', help='use DGCNN' )
parser.add_argument('--use_SSG', action='store_true', help='use SSG' )
parser.add_argument('--use_MSG', action='store_true', help='use MSG' )
parser.add_argument('--norm', action='store_true', help='use normalized data' )
parser.add_argument('--no_fnorm', action='store_true', help='no_fnorm' )
parser.add_argument('--use_dense2', action='store_true', help='use dense' )
parser.add_argument('--use2', action='store_true', help='use model2' )
parser.add_argument('--id_skip',type=int, default=0, help='use dense' )
parser.add_argument('--scale', action='store_true', help='enable scale' )
parser.add_argument('--step', action='store_true', help='enable step lrRate' )
parser.add_argument('--rotate', action='store_true', help='enable rotate' )
parser.add_argument('--bg', type=int, default=0, help='enable background when bg==1, enable romp 3d when bg==2' )
parser.add_argument('--light', action='store_true', help='enable light model' )
parser.add_argument('--jitter', action='store_true', help='enable graph jitter' )
parser.add_argument('--D2', action='store_true', help='enable graph jitter' )
parser.add_argument('--no_se', action='store_true', help='enable graph jitter' )
parser.add_argument('--fp16', action='store_true', help='enable fp16' )
parser.add_argument('--class_sampler', type=int, default=0, help='enable class sampler' )
parser.add_argument('--erase', type=float, default=0)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--no_xyz', action='store_true')
parser.add_argument('--ASPP', type=int, default=0)
parser.add_argument('--final_bn', action='store_true')
parser.add_argument('--efficient', action='store_true')
parser.add_argument('--train_all', action='store_true')
parser.add_argument('--pre_act', action='store_true')
parser.add_argument('--balance', action='store_true', help='balance sample' )
parser.add_argument('--slim', type=float, default=0.5)
parser.add_argument('--layer_drop', type=float, default=0)
parser.add_argument('--norm_layer', type=str, default='bn')
parser.add_argument('--drop_connect_rate', type=float, default=0.0)
parser.add_argument('--channel', type=int, default=6)
parser.add_argument('--warm_epoch', default=-1, type=int, help='stride')
parser.add_argument('--shuffle', default=-1, type=int, help='shuffle channel')
parser.add_argument('--labelsmooth', action='store_true', help='use label smooth' )
parser.add_argument('--sync_bn', action='store_true', help='use label smooth' )
parser.add_argument('--sam', action='store_true', help='use sam optimizer' )
parser.add_argument('--lookahead', action='store_true', help='use lookahead optimizer' )
parser.add_argument('--wa', action='store_true', help='use weight average' )
parser.add_argument('--cg', action='store_true', help='use context gating before aggregation' )
parser.add_argument('--twins', action='store_true', help='use barlow-twins loss' )
parser.add_argument('--amsgrad', action='store_true', help='use amsgrad for adam / adamW' )
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
parser.add_argument('--nce', action='store_true', help='use nce loss' )
parser.add_argument('--wa_start', default=0.9, type=float, help='use weight average, when to start' )
opt = parser.parse_args()


if not opt.pretrained_name == 'None':
    popt = parser.parse_args()

    config_path = os.path.join('./snapshot',popt.pretrained_name,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    popt.slim = config['slim']
    popt.use_dense = config['use_dense']
    popt.k = config['k']
    popt.class_num = config['class_num']
    popt.channel = config['channel']
    popt.init_points= config['init_points']
    popt.use_dense2 = config['use_dense2']
    popt.norm = config['norm']
    popt.gpu_ids = config['gpu_ids']
    popt.npart = config['npart']
    popt.id_skip = config['id_skip']
    popt.feature_dims = config['feature_dims']
    if 'pre_act' in config:
        popt.pre_act = config['pre_act']
    else: 
        popt.pre_act = False
    if 'norm_layer' in config:
        popt.norm_layer = config['norm_layer']
    else:
        popt.norm_layer = 'bn'
    if 'stride' in config:
        popt.stride = config['stride']
    else:
        popt.stride = 2
    if 'layer_drop' in config:
        popt.layer_drop = config['layer_drop']
    else:
        popt.layer_drop = 0
    if 'num_conv' in config:
        popt.num_conv = config['num_conv']
    else:
        popt.num_conv = 1
    if 'efficient' in config:
        popt.efficient = config['efficient']

    if 'final_bn' in config:
        popt.final_bn = config['final_bn']

    if 'shuffle' in config:
        popt.shuffle = config['shuffle']

    if 'twins' in config:
        popt.twins = config['twins']
        popt.nce = config['nce']

    if 'circle' in config:
        popt.circle = config['circle']
        
    if 'arcface' in config:
        popt.arcface = config['arcface']
        popt.cosface = config['cosface']
        popt.contrast = config['contrast']
        popt.triplet = config['triplet']
        popt.lifted = config['lifted']
        popt.sphere = config['sphere']

    if 'gem' in config:
        popt.gem = config['gem']

    if 'ASPP' in config:
        popt.ASPP = config['ASPP']

    if 'cg' in config:
        popt.cg = config['cg']

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
        pin_memory=True,
        shuffle=True,
        drop_last=True)
CustomTestDataLoader = CustomDataLoader
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def calculate_loss(model, criterion, data, label): 
    loss_twins = 0.0
    loss_circle = 0.0
    loss = 0.0
    xyz = data[:,:,0:3].contiguous()
    rgb = data[:,:,3:].contiguous()
    logits = model(xyz.detach(), rgb.detach(), istrain=True)
    #loss = compute_loss(logits, label)
    return_feature = opt.twins or opt.nce or opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
    if return_feature:
        logits, feature = logits

    if opt.npart>1:
        for i in range(0, opt.npart+1):
            #if i==0:
            #    loss = criterion(logits[0], label)
            #else:
            loss += criterion(logits[i], label)
            if opt.circle:
                loss_circle += criterion_circle(*convert_label_to_similarity(feature[i], label))/opt.batch_size
        loss += loss_circle
    else:
        if opt.twins:
            feature = feature.t()
        if return_feature:
            if not opt.no_fnorm:
                feature = L2norm(feature)
        loss = criterion(logits, label)
        if opt.circle:
            loss_circle = criterion_circle(*convert_label_to_similarity(feature, label))/opt.batch_size
            loss += loss_circle
        if opt.arcface:
            loss +=  criterion_arcface(feature, label)/opt.batch_size
        if opt.cosface:
            loss +=  criterion_cosface(feature, label)/opt.batch_size
        if opt.triplet:
            hard_pairs = miner(feature, label)
            loss +=  criterion_triplet(feature, label, hard_pairs) #/now_batch_size
        if opt.lifted:
            loss +=  criterion_lifted(feature, label) #/now_batch_size
        if opt.contrast:
            loss +=  criterion_contrast(feature, label) #/now_batch_size
        if opt.sphere:
            loss +=  criterion_sphere(feature, label)/opt.batch_size      

        if opt.twins:
            sim1 = torch.mm(feature*torch.exp(model.module.logit_scale), torch.t(feature))
            sim2 = sim1.t()
            sim_label = torch.arange(sim1.size(0), device="cuda").detach()
            loss_twins = F.cross_entropy(sim1, sim_label) + F.cross_entropy(sim2, sim_label)
            loss += 0.5*loss_twins

    if epoch<opt.warm_epoch:
        opt.warm_up = min(1.0, opt.warm_up + 0.9 / opt.warm_iteration)
        loss *= opt.warm_up
    return loss, loss_twins, loss_circle, logits, opt.warm_up

market_data = Market3D(opt.dataset_path, flip=opt.flip, slim=opt.slim, norm = opt.norm, scale = opt.scale, erase=opt.erase, rotate = opt.rotate, channel = opt.channel, bg=opt.bg, D2 = opt.D2, class_sampler = opt.class_sampler)
opt.class_num = market_data.train().class_num

if opt.labelsmooth:
    criterion = CrossEntropyLabelSmooth()
if opt.circle:
    criterion_circle = CircleLoss(m=0.25, gamma=opt.gamma)
if opt.arcface:
    criterion_arcface = losses.ArcFaceLoss(num_classes=opt.class_num, embedding_size=512)
if opt.cosface: 
    criterion_cosface = losses.CosFaceLoss(num_classes=opt.class_num, embedding_size=512)
if opt.triplet:
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
if opt.lifted:
    criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
if opt.contrast: 
    criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
if opt.sphere:
    criterion_sphere = losses.SphereFaceLoss(num_classes=opt.class_num, embedding_size=512, margin=4)

def train(model, optimizer, scheduler, train_loader, dev, epoch):

    model.train()
    total_loss = 0
    total_loss_twins = 0
    total_loss_circle = 0
    num_batches = 0
    total_correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()

    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev).squeeze().long()
            optimizer.zero_grad()

            loss, loss_twins, loss_circle, logits, warm_up = calculate_loss(model, criterion, data, label)

            if opt.npart>1:
                logit_sum = logits[0].detach()
                for i in range(1, opt.npart+1):
                    logit_sum += logits[i].detach()
                _, preds = logit_sum.max(1)
            else:
                _, preds = logits.max(1)

            del logits, data

            if opt.fp16: # we use optimier to backward loss
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if opt.sam: # second forward-backward pass
                optimizer.first_step(zero_grad=True)
                disable_bn(model)
                loss, _, _, _, warm_up = calculate_loss(model, criterion, data, label)
                if opt.fp16: # we use optimier to backward loss
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.second_step(zero_grad=True)
                enable_bn(model)
            else:
                optimizer.step()


            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss.item()
            total_correct += correct

            show = {
                #'Loss': '%.5f' % loss,
                'WarmUp':'%.4f' % (warm_up),
                'AvgLoss': '%.4f' % (total_loss / num_batches),
                #'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.4f' % (total_correct / count)}
            if opt.twins:
                total_loss_twins += loss_twins.item()
                show['Twins'] = '%.4f'%(total_loss_twins/num_batches)

            if opt.circle:
                total_loss_circle += loss_circle.item()
                show['Circle'] = '%.4f'%(total_loss_circle/num_batches)

            tq.set_postfix(show)
            del label, loss, loss_twins, loss_circle, warm_up
        y_loss['train'].append(total_loss / num_batches)
        y_err['train'].append(1.0-total_correct / count)


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
                return_feature = opt.twins or opt.nce or opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
                if return_feature:
                    logits , features = logits
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




if opt.balance and not opt.class_sampler: # when we use class_sampler, we do not use balance
    if opt.train_all:
        dataset_train = market_data.train_all()
    else:
        dataset_train = market_data.train()
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler = sampler,
        pin_memory=True,
        drop_last=True)


train_loader = CustomDataLoader(market_data.train())
if opt.train_all:
    train_loader = CustomDataLoader(market_data.train_all())
valid_loader = CustomDataLoader(market_data.valid())

dataset_sizes = {}
dataset_sizes['train'] = market_data.train().img_num if not opt.train_all else market_data.train_all().img_num
dataset_sizes['val'] = market_data.valid().img_num

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init
opt.warm_up = 0.1
opt.warm_iteration = round(dataset_sizes['train']/opt.batch_size)*opt.warm_epoch # first 5 epoch
total_iteration = round(dataset_sizes['train']/opt.batch_size)*opt.num_epochs

return_feature = opt.twins or opt.nce or opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
if opt.use_dense and not opt.efficient:
    if not opt.pretrained_name == 'None':
        return_feature_p = popt.twins or popt.nce or popt.arcface or popt.cosface or popt.circle or popt.triplet or popt.contrast or popt.lifted or popt.sphere
        pretrained_model = Model_dense(popt.k, popt.feature_dims, [512], output_classes=popt.class_num, init_points = popt.init_points, input_dims=3, dropout_prob=popt.droprate, npart = popt.npart, id_skip=popt.id_skip, drop_connect_rate = popt.drop_connect_rate, res_scale = popt.res_scale, light = popt.light, cluster = popt.cluster, conv=popt.conv, use_xyz = not popt.no_xyz, use_se = not popt.no_se, graph_jitter = popt.jitter, pre_act= popt.pre_act, norm = popt.norm_layer, stride = popt.stride, layer_drop = popt.layer_drop )

    model = Model_dense(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, use_se = not opt.no_se, graph_jitter = opt.jitter, pre_act = opt.pre_act, norm = opt.norm_layer , stride = opt.stride, layer_drop = opt.layer_drop )
elif opt.use2: 
    model = ModelE_dense2(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, use_se = not opt.no_se, graph_jitter = opt.jitter, pre_act = opt.pre_act, norm = opt.norm_layer , stride = opt.stride, layer_drop = opt.layer_drop, num_conv = opt.num_conv, shuffle = opt.shuffle)
elif opt.efficient:
    if not opt.pretrained_name == 'None':
        pretrained_model = ModelE_dense(popt.k, popt.feature_dims, [512], output_classes=popt.class_num, init_points = popt.init_points, input_dims=3, dropout_prob=popt.droprate, npart = popt.npart, id_skip=popt.id_skip, drop_connect_rate = popt.drop_connect_rate, res_scale = popt.res_scale, light = popt.light, cluster = popt.cluster, conv=popt.conv, use_xyz = not popt.no_xyz, use_se = not popt.no_se, graph_jitter = popt.jitter, pre_act= popt.pre_act, norm = popt.norm_layer, stride = popt.stride, layer_drop = popt.layer_drop, num_conv = popt.num_conv, temp = return_feature_p, gem = popt.gem , ASPP = popt.ASPP, cg = popt.cg)

    model = ModelE_dense(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, use_se = not opt.no_se, graph_jitter = opt.jitter, pre_act = opt.pre_act, norm = opt.norm_layer , stride = opt.stride, layer_drop = opt.layer_drop, num_conv = opt.num_conv, temp = return_feature, gem = opt.gem , ASPP = opt.ASPP, cg = opt.cg)
elif opt.use_dense2: 
    model = Model_dense2(opt.k, opt.feature_dims, [512],  output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate, npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate=opt.drop_connect_rate, res_scale =opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, graph_jitter = opt.jitter, pre_act = opt.pre_act, norm = opt.norm_layer , stride = opt.stride, layer_drop = opt.layer_drop)
elif opt.use_DGCNN:
    model = DGCNN( 20, [64,128,256,512], [512,512], output_classes=opt.class_num,  input_dims=3, dropout_prob=opt.droprate )
elif opt.use_SSG:
    model = PointNet2SSG(output_classes=opt.class_num, init_points = 512, input_dims=3, dropout_prob=opt.droprate, use_xyz = not opt.no_xyz, temp = return_feature )
elif opt.use_MSG:
    model = PointNet2MSG(output_classes=opt.class_num, init_points = 512, input_dims=3, dropout_prob=opt.droprate, use_xyz = not opt.no_xyz, temp = return_feature )
else:
    model = Model(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, dropout_prob=opt.droprate , npart = opt.npart, id_skip=opt.id_skip, drop_connect_rate = opt.drop_connect_rate, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv, use_xyz = not opt.no_xyz, graph_jitter = opt.jitter, pre_act = opt.pre_act, norm = opt.norm_layer , stride = opt.stride, layer_drop = opt.layer_drop)

if not opt.pretrained_name == 'None':
    model_path = './snapshot/'+opt.pretrained_name+'/model_last.pth'
    pretrained_model = torch.nn.DataParallel(pretrained_model, device_ids=popt.gpu_ids)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.load_state_dict(torch.load(model_path, map_location=dev))
    #print(pretrained_model)
    pretrained_model.module.proj_output = model.proj_output.cuda()
    model = pretrained_model.module

print(model)
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print('Number of training parameters: %.2f M'% (params/1e6) )
query_loader = CustomTestDataLoader(market_data.query())
batch0,label0 = next(iter(query_loader))
batch0 = batch0[0].unsqueeze(0)
print(batch0.shape)
macs, params = get_model_complexity_info(model.cuda(), batch0.cuda(), ((round(6890*opt.slim), 3)  ), as_strings=True, print_per_layer_stat=False, verbose=True)
#print(macs)
del(batch0)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if opt.sync_bn:
    import apex
    print("using apex synced BN")
    model = apex.parallel.convert_syncbn_model(model)

optimizer = optim.SGD(model.parameters(), lr=opt.lrRate, momentum=0.9, nesterov=True, weight_decay=1e-4)
if opt.sam:
    optimizer = SAM(model.parameters(), base_optimizer=optim.SGD, lr=opt.lrRate, momentum=0.9, nesterov=True, weight_decay=1e-4)

if opt.adam:
    optimizer = optim.Adam(model.parameters(), lr = opt.lrRate, weight_decay=opt.wd, amsgrad = opt.amsgrad)
    if opt.sam:
        optimizer = SAM(model.parameters(), base_optimizer=optim.Adam, lr=opt.lrRate, weight_decay=opt.wd, amsgrad = opt.amsgrad)
    elif opt.lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

if opt.adamW:
    optimizer = optim.AdamW(model.parameters(), lr = opt.lrRate, weight_decay=opt.wd, amsgrad = opt.amsgrad)
    if opt.sam:
        optimizer = SAM(model.parameters(), base_optimizer=optim.AdamW, lr = opt.lrRate, weight_decay=opt.wd, amsgrad = opt.amsgrad) 
    elif opt.lookahead:
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.num_epochs, eta_min=0.01*opt.lrRate)

if opt.step:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.num_epochs*0.8, opt.num_epochs*0.95], gamma=0.1)

if opt.restart:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.num_epochs//16, T_mult=2, eta_min=0.01*opt.lrRate)
    opt.num_epochs = opt.num_epochs//16 * (1+2+4+8)
    print('new num_epoch %d'%opt.num_epochs)

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

if opt.fp16:
    model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")

#model = model.to(dev)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
if opt.load_model_path:
    model.load_state_dict(torch.load(opt.load_model_path, map_location=dev))

wa_flag = opt.wa
wa_start = opt.wa_start*opt.num_epochs
inner_loop = math.ceil(dataset_sizes['train']/opt.class_num /opt.class_sampler) if opt.class_sampler else 1

print('Total Iteration: %d'%total_iteration)
for epoch in range(opt.num_epochs):
    print('Epoch #%d / %d' %(epoch, opt.num_epochs))

    # moving average
    if opt.wa and wa_flag and epoch >= wa_start:
        wa_flag = False
        swa_model = swa_utils.AveragedModel(model)
        swa_utils.update_bn(train_loader, swa_model, device = dev )
        print('start weight avg')

    if not opt.train_all:
        valid_acc = evaluate(model, valid_loader, dev)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_model_path + '/model_best.pth')
        print('Current validation acc: %.5f (best: %.5f)' % (
            valid_acc, best_valid_acc))
    if epoch%50 == 9 or epoch == opt.num_epochs-1 : # save model every 50 epoch
        torch.save(model.state_dict(), save_model_path + '/model_last.pth')
    for inner in range(inner_loop):
        train(model, optimizer, scheduler, train_loader, dev, epoch)
    scheduler.step()
    if not opt.train_all:
        draw_curve(epoch)
    if opt.wa and epoch >= wa_start:
        swa_model.update_parameters(model)
        if epoch == opt.num_epochs-1 : # save last avg module
            torch.save(swa_model.module.state_dict(), save_model_path + '/model_average_nobn.pth')
            swa_utils.update_bn( train_loader, swa_model, device = dev)
            torch.save(swa_model.module.state_dict(), save_model_path + '/model_average.pth')
