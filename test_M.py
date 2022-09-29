import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from market3d import Market3D
from model import Model,Model_dense,Model_dense2
from model_efficient import ModelE_dense
from model_efficient2 import ModelE_dense2
from dgl.data.utils import download, get_download_dir
import torch.backends.cudnn as cudnn
from functools import partial
import tqdm
import urllib
import os
import argparse
import scipy.io
import yaml
import numpy as np
from ptflops import get_model_complexity_info
from DGCNN import DGCNN
from pointnet2_model import PointNet2SSG, PointNet2MSG
import swa_utils
from utils import L2norm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--feature_dims',default=[64,128,256,512], type=list,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which-epoch', type=str, default='last')
parser.add_argument('--dataset-path', type=str, default='./2DMarket/')
parser.add_argument('--load-model-path', type=str, default='./snapshot/')
parser.add_argument('--name', type=str, default='b24_lr2')
parser.add_argument('--cluster', type=str, default='xyz')
parser.add_argument('--conv', type=str, default='EdgeConv')
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--init_points', type=int, default=512)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--class-num', type=int, default=751)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--use_DGCNN', action='store_true', help='use DGCNN' )
parser.add_argument('--use2', action='store_true', help='use model2' )
parser.add_argument('--use_SSG', action='store_true', help='use SSG' )
parser.add_argument('--use_MSG', action='store_true', help='use MSG' )
parser.add_argument('--npart', type=int, default=1)
parser.add_argument('--channel', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=48)
parser.add_argument('--resume', action='store_true', help='resume training' )
parser.add_argument('--flip', action='store_true', help='flip' )
parser.add_argument('--id_skip',type=int, default=0, help='use dense' )
parser.add_argument('--use_dense', action='store_true', help='use dense' )
parser.add_argument('--norm', action='store_true', help='use normalized input' )
parser.add_argument('--bg', type=int, default=0, help='use background' )
parser.add_argument('--light', action='store_true', help='use light model' )
parser.add_argument('--no_se', action='store_true', help='use light model' )
parser.add_argument('--final_bn', action='store_true', help='add bn' )
parser.add_argument('--slim', type=float, default=0.3 )
parser.add_argument('--layer_drop', type=float, default=0.0 )
parser.add_argument('--norm_layer', type=str, default='bn')
parser.add_argument('--rotate', type=int, default=0)
parser.add_argument('--no_xyz', action='store_true')
parser.add_argument('--ASPP', type=int, default=0)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--pre_act', action='store_true')
parser.add_argument('--D2', action='store_true')
parser.add_argument('--efficient', action='store_true')
parser.add_argument('--wa', action='store_true')
parser.add_argument('--gem', action='store_true')
parser.add_argument('--cg', action='store_true', help='use gc before aggregation' )
parser.add_argument('--twins', action='store_true', help='use barlow-twins loss' )
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
parser.add_argument('--nce', action='store_true', help='use nce loss' )
parser.add_argument('--update_bn', action='store_true')
parser.add_argument('--res_scale', type=float, default=1.0 )
opt = parser.parse_args()

config_path = os.path.join('./snapshot',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
#assert not ('MSMT' in opt.name)
opt.slim = config['slim'] 
print('slim: %.2f:'%opt.slim)
opt.use_dense = config['use_dense']
opt.k = config['k']
opt.class_num = config['class_num']
opt.channel = config['channel']
opt.init_points= config['init_points'] 
opt.use_dense2 = config['use_dense2']
opt.norm = config['norm']
opt.npart = config['npart']
opt.id_skip = config['id_skip']
opt.feature_dims = config['feature_dims']
opt.light = config['light']

if 'use2' in config:
    opt.use2 = config['use2']

if 'res_scale' in config:
    opt.res_scale = config['res_scale']

if 'bg' in config:
    opt.bg = config['bg']

print('bg:%d'%opt.bg)

if 'cluster' in config:
    opt.cluster = config['cluster']

if 'use_DGCNN' in config:
    opt.use_DGCNN = config['use_DGCNN']
if 'use_SSG' in config:
    opt.use_SSG = config['use_SSG']
    opt.use_MSG = config['use_MSG']

if 'conv' in config:
    opt.conv = config['conv']

if 'no_se' in config:
    opt.no_se = config['no_se']

if 'no_xyz' in config:
    opt.no_xyz = config['no_xyz']
if 'D2' in config:
    opt.D2 = config['D2']

if 'pre_act' in config:
    opt.pre_act = config['pre_act']

if 'norm_layer' in config:
    opt.norm_layer = config['norm_layer']
else:
    opt.norm_layer = 'bn'

if 'stride' in config:
    opt.stride = config['stride']
else:
    opt.stride = 2

if 'layer_drop' in config:
    opt.layer_drop = config['layer_drop']
else:
    opt.layer_drop = 0

if 'num_conv' in config:
    opt.num_conv = config['num_conv']
else:
    opt.num_conv = 1

if 'efficient' in config:
    opt.efficient = config['efficient']

if 'final_bn' in config:
    opt.final_bn = config['final_bn']

if 'shuffle' in config:
    opt.shuffle = config['shuffle']

if 'twins' in config:
    opt.twins = config['twins']
    opt.nce = config['nce']

if 'circle' in config:
    opt.circle = config['circle']

if 'triplet' in config:
    opt.triplet = config['triplet']
    opt.arcface = config['arcface']
    opt.cosface = config['cosface']
    opt.sphere = config['sphere']
    opt.lifted = config['lifted']
    opt.contrast = config['contrast']
        
if 'gem' in config:
    opt.gem = config['gem']

if 'ASPP' in config:
    opt.ASPP = config['ASPP']

if 'cg' in config:
    opt.cg = config['cg']

#if type(opt.feature_dims)==:
#   str_features = opt.feature_dims.split(',')
#   features = []
#   for feature in str_features:
#        feature = int(feature)
#        features.append(feature)
#    opt.feature_dims = features

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

# set gpu ids
if len(opt.gpu_ids)>0:
    cudnn.enabled = True
    cudnn.benchmark = True

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

def extract_feature(model, test_loader, dev, rotate = 0):
    model.eval()

    total_correct = 0
    count = 0
    features = torch.FloatTensor()
    with tqdm.tqdm(test_loader, ascii=True) as tq:
        for data, label in tq: # n,6890,6
            num_examples = label.shape[0]
            n, c, l = data.size()
            if opt.npart>1:
                ff = torch.FloatTensor(n, 512*(opt.npart+1) ).zero_().cuda()
            else:
                ff = torch.FloatTensor(n, 512 ).zero_().cuda()
            data, label = data.to(dev), label.to(dev).squeeze().long()
            xyz = data[:,:,0:3].contiguous()
            rgb = data[:,:,3:].contiguous()
            if rotate ==  90:
                xyz_clone = xyz.clone()
                xyz[:,:,0] = xyz_clone[:,:,1]
                xyz[:,:,1] = xyz_clone[:,:,0]
            elif rotate == 180:
                xyz[:,:,1] *= -1
            output = model(xyz, rgb, istrain=False)
            return_feature = opt.twins or opt.nce or opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere
            if return_feature:
                output = output[0]

            if opt.npart>1:
                for i in range(opt.npart+1):
                    start = 512*i
                    end = 512*i + 512
                    ff[:, start:end] += L2norm(output[i])
            else:
                ff += output
            #flip
            #xyz[:,:,0] *= -1
            #scale
            xyz *=1.1
            output = model(xyz, rgb, istrain=False)

            if return_feature:
                output = output[0]

            if opt.npart>1:
                for i in range(opt.npart+1):
                    start = 512*i
                    end = 512*i + 512
                    ff[:, start:end] += L2norm(output[i])
            else:
                ff += output

            ff = L2norm(ff)
            #fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            #ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


market_data = Market3D(opt.dataset_path, flip=False, slim=opt.slim, norm =opt.norm, erase=0, channel = opt.channel, bg = opt.bg, D2 = opt.D2)

train_loader = CustomDataLoader(market_data.train_all())
query_loader = CustomDataLoader(market_data.query())
gallery_loader = CustomDataLoader(market_data.gallery())

gallery_path = market_data.gallery().imgs
query_path = market_data.query().imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

return_feature = opt.twins or opt.nce or opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.lifted or opt.sphere

if opt.use_dense and not opt.efficient:
    model = Model_dense(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, npart = opt.npart, id_skip = opt.id_skip, res_scale = opt.res_scale, light=opt.light, cluster = opt.cluster, conv = opt.conv,  use_xyz = not opt.no_xyz, use_se = not opt.no_se, pre_act = opt.pre_act, norm = opt.norm_layer, stride = opt.stride, layer_drop = opt.layer_drop)
elif opt.use2:
    model = ModelE_dense2(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, npart = opt.npart, id_skip = opt.id_skip, res_scale = opt.res_scale, light=opt.light, cluster = opt.cluster, conv = opt.conv,  use_xyz = not opt.no_xyz, use_se = not opt.no_se, pre_act = opt.pre_act, norm = opt.norm_layer, stride = opt.stride, layer_drop = opt.layer_drop, num_conv = opt.num_conv, shuffle = opt.shuffle)
elif opt.efficient:
    model = ModelE_dense(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, npart = opt.npart, id_skip = opt.id_skip, res_scale = opt.res_scale, light=opt.light, cluster = opt.cluster, conv = opt.conv,  use_xyz = not opt.no_xyz, use_se = not opt.no_se, pre_act = opt.pre_act, norm = opt.norm_layer, stride = opt.stride, layer_drop = opt.layer_drop, num_conv = opt.num_conv,  temp = return_feature , gem= opt.gem, ASPP = opt.ASPP, cg =opt.cg)
elif opt.use_dense2:
    model = Model_dense2(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, npart = opt.npart, id_skip = opt.id_skip, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv,  use_xyz = not opt.no_xyz, pre_act = opt.pre_act)
elif opt.use_DGCNN:
    model = DGCNN( 20, [64,128,256,512], [512,512], output_classes=opt.class_num,  input_dims=3)
elif opt.use_SSG:
    model = PointNet2SSG(output_classes=opt.class_num, init_points = 512, input_dims=3, use_xyz = not opt.no_xyz, temp = return_feature)
elif opt.use_MSG:
    model = PointNet2MSG(output_classes=opt.class_num, init_points = 512, input_dims=3, use_xyz = not opt.no_xyz, temp = return_feature)
else:
    model = Model(opt.k, opt.feature_dims, [512], output_classes=opt.class_num, init_points = opt.init_points, input_dims=3, npart = opt.npart, id_skip = opt.id_skip, res_scale = opt.res_scale, light = opt.light, cluster = opt.cluster, conv=opt.conv,  use_xyz = not opt.no_xyz, pre_act = opt.pre_act, norm = opt.norm_layer, stride = opt.stride, layer_drop = opt.layer_drop)

#model = model.to(dev)
model_path = opt.load_model_path+opt.name+'/model_%s.pth'%opt.which_epoch

try:
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.proj_output = nn.Sequential()
    model.classifier = nn.Sequential()
except:
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.module.proj_output = nn.Sequential()
    model.module.classifier = nn.Sequential()
    if opt.npart>1:
        for i in range(opt.npart+1):
            model.module.proj_outputs[i] = nn.Sequential()


print(model)
print(model_path)

batch0,label0 = next(iter(query_loader))
batch0 = batch0[0].unsqueeze(0)
print(batch0.shape)
macs, params = get_model_complexity_info(model, batch0, ((round(6890*opt.slim), 3)  ), as_strings=True, print_per_layer_stat=False, verbose=True)
#print(macs)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#print('Number of parameters: %.2f M'% (params/1e6) )

if not os.path.exists('./snapshot/'):
    os.mkdir('./snapshot/')
save_model_path = './snapshot/' + opt.name
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

if opt.update_bn:
    with torch.no_grad():
        swa_utils.update_bn( train_loader, model, device = 'cuda')

# Extract feature
with torch.no_grad():
    query_feature = extract_feature(model, query_loader, dev, rotate = opt.rotate)
    gallery_feature = extract_feature(model, gallery_loader, dev, rotate = opt.rotate)

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = './snapshot/%s/result.txt'%opt.name
os.system('python evaluate_gpu.py | tee -a %s'%result)
