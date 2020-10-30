import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from modelnet import ModelNet
from dgl.data.utils import download, get_download_dir
import torch.backends.cudnn as cudnn
from functools import partial
import tqdm
import urllib
import os
import argparse
import yaml
from model import Model, Model_dense, Model_dense2
from utils import CrossEntropyLabelSmooth

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='../modelnet40-sampled-2048.h5')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--name', type=str, default='basic')
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--use_dense', action='store_true', help='use dense' )
parser.add_argument('--adam', action='store_true', help='use dense' )
parser.add_argument('--train_all', action='store_true', help='use dense' )
parser.add_argument('--labelsmooth', action='store_true', help='use dense' )
parser.add_argument('--warm_epoch', default=5, type=int, help='stride')
parser.add_argument('--cluster', default='xyzrgb', type=str, help='stride')
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
data_filename = 'modelnet40-sampled-2048.h5'
local_path = args.dataset_path or os.path.join(get_download_dir(), data_filename)

if not os.path.exists(local_path):
    download('https://data.dgl.ai/dataset/modelnet40-sampled-2048.h5', local_path)

cudnn.enabled = True
cudnn.benchmark = True

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

def evaluate(model, test_loader, dev):
    model.eval()

    total_correct = 0
    count = 0
    mean_class = 0
    class_correct = torch.zeros(40).cuda()
    class_label = torch.zeros(40).cuda()
    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq: # 1024,3
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = model(data, data)
                #scale
                logits += model(1.1*data, 1.1*data)
                logits += model(0.9*data, 0.9*data)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                for i in range(label.shape[0]):
                    class_label[label[i]] +=1
                    if preds[i] == label[i]:
                        class_correct[label[i]] +=1

            for i in range(40):
                mean_class += class_correct[i]/class_label[i]
            tq.set_postfix({
                'MeanClass': '%.4f' % (mean_class / 40),
                'AvgAcc': '%.4f' % (total_correct / count)})

    return mean_class / 40, total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = Model(20, [64, 64, 128, 256], [512, 512, 256], 40)

model = Model_dense(20, [64,128,256,512], [512], output_classes=40, init_points = 768, input_dims=3, dropout_prob=0.7, cluster=args.cluster)

model = model.to(dev)
model = nn.DataParallel(model)
if args.load_model_path:
    model.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
if args.adam:
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.01*args.lr)

modelnet = ModelNet(local_path, 1024)

train_loader = CustomDataLoader(modelnet.train())
if args.train_all:
    train_loader = CustomDataLoader(modelnet.train_all())
valid_loader = CustomDataLoader(modelnet.valid())
test_loader = CustomDataLoader(modelnet.test())

dataset_sizes = {}
dataset_sizes['train'] = modelnet.n_train
dataset_sizes['val'] = modelnet.n_valid

best_valid_acc = 0
best_test_acc = 0

save_model_path = './snapshot/' + args.name + '/model_last.pth'
try:
    model.load_state_dict(torch.load(save_model_path))
except:
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(save_model_path))

model.cuda()
print(model)

mtest_acc, test_acc = evaluate(model, test_loader, dev)
print(mtest_acc, test_acc)
