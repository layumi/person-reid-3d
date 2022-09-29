from torchvision import datasets
import os
import numpy as np
import random
import torch
import open3d as o3d
import random
from torch.utils import data
import dgl

class Market3D(object):
    def __init__(self, path, flip=False, slim=1.0, scale=False, norm=False, erase=0, rotate = False, channel=6, bg = False, D2 = False, class_sampler = 0 ):
        self.path = path
        self.flip = flip
        self.slim = slim
        self.scale = scale
        self.norm = norm
        self.erase = erase
        self.rotate = rotate
        self.channel = channel
        self.bg = bg
        self.D2 = D2
        self.class_sampler = class_sampler

    def train(self):
        return Market3DFolder(self.path +'/train', flip=self.flip, slim = self.slim, scale = self.scale, norm = self.norm, erase = self.erase, rotate=self.rotate, channel = self.channel, bg =self.bg, D2=self.D2, class_sampler = self.class_sampler)

    def train_all(self):
        return Market3DFolder(self.path +'/train_all', flip=self.flip, slim = self.slim, scale = self.scale, norm = self.norm, erase = self.erase, rotate=self.rotate, channel = self.channel, bg =self.bg, D2=self.D2, class_sampler = self.class_sampler)

    def valid(self):
        return Market3DFolder(self.path+'/val', slim = self.slim, norm=self.norm, erase=0, channel = self.channel, bg=self.bg, D2 = self.D2)

    def query(self):
        return Market3DFolder(self.path+'/query', slim = self.slim, norm=self.norm, erase=0, channel = self.channel, bg=self.bg, D2 = self.D2)

    def gallery(self):
        return Market3DFolder(self.path+'/gallery', slim = self.slim, norm=self.norm, erase=0, channel = self.channel, bg=self.bg, D2 = self.D2)

    def build_human_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(6890)
        with open(self.path.replace('2D','3D') +'/train/0002/0002_c1s1_000451_03.jpg.obj','r') as f:
            for line in f:
                if not line[0] == 'f':
                    continue
                face = line.split(' ')
                g.add_edge(int(face[1])-1, int(face[2])-1)
                g.add_edge(int(face[1])-1, int(face[3])-1)
                g.add_edge(int(face[2])-1, int(face[1])-1)
                g.add_edge(int(face[2])-1, int(face[3])-1)
                g.add_edge(int(face[3])-1, int(face[1])-1)
                g.add_edge(int(face[3])-1, int(face[2])-1)
        return g


class Market3DFolder(datasets.ImageFolder):
    def __init__(self, root, flip=False, slim=1.0, scale = False, norm = False, erase=0, rotate=False, channel = 6, transform=None, bg = False, D2 = False, class_sampler = 0):
        super(Market3DFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        if bg==1:
            objs = [s[0].replace('2DMarket','3DMarket+bg').replace('2DDuke','3DDuke+bg').replace('2DMSMT','3DMSMT+bg').replace('2DCUHK','3DCUHK+bg').replace('2DVIP','3DVIP+bg')+'.obj' for s in self.samples]
        elif bg ==2:
            objs = [s[0].replace('2DMarket','3DMarket_ROMP').replace('2DDuke','3DDuke_ROMP').replace('2DMSMT','3DMSMT_ROMP').replace('2DCUHK','3DCUHK_ROMP').replace('2DVIP','3DVIP_ROMP')+'.obj' for s in self.samples]
        else:
            objs = [s[0].replace('2DMarket','3DMarket').replace('2DDuke','3DDuke').replace('2DMSMT','3DMSMT+bg').replace('2DCUHK','3DCUHK+bg').replace('2DVIP','3DVIP+bg')+'.obj' for s in self.samples]

        self.targets = targets
        self.objs = objs
        self.flip = flip
        self.slim = slim
        self.scale = scale
        self.norm = norm
        self.erase = erase
        self.rotate = rotate
        self.channel = channel
        self.img_num = len(self.samples)
        self.class_num = len(np.unique(targets))
        self.img = self.samples
        self.root = root
        self.D2 = D2
        self.class_sampler = class_sampler

    def __len__(self):
        if self.class_sampler:
            return self.class_num * self.class_sampler
        return self.img_num

    def __getitem__(self, index):
        if self.class_sampler: # class index -> sample index
            index = index % self.class_num
            ava_index = np.argwhere(self.targets == index)
            rand_index = np.random.permutation(len(ava_index))[0]
            index = ava_index[rand_index][0]
        path, target = self.samples[index]
        path3d = self.objs[index]
        mesh = o3d.io.read_triangle_mesh(path3d)
        obj = np.asarray(mesh.vertices, dtype=np.float32)
        obj -= np.mean(obj, axis=0)
        if self.flip and random.random() < 0.5:
            obj[:,0] *= -1 
        obj_color = np.asarray(mesh.vertex_colors, dtype=np.float32)

        obj = np.concatenate((obj, obj_color), axis=1)
        if self.slim<1.0:
            v_num = obj.shape[0]
            rgb_mean  = np.mean( obj[:,3:], axis=1)
            blank_point = np.argwhere(np.abs(rgb_mean-1.0)<=0.00001)
            not_blank_point = np.argwhere(np.abs(rgb_mean-1.0)>0.00001)
            obj[blank_point, 3:] = 0.5 # reset to 0.5
            return_point = round(v_num*self.slim)
            if return_point<=len(not_blank_point): 
                if 'train' in self.root and random.random() < 0.5: 
                # random select points 
                    in_selected = np.random.permutation(len(not_blank_point)-1)[:return_point]
                else:
                # linear select points 
                    in_selected = np.linspace(0, len(not_blank_point)-1, num= return_point, dtype=int)
                selected = not_blank_point[in_selected]
            else: 
                # all the color inputs with some blank points
                out_selected = np.linspace(0, len(blank_point)-1, num= return_point-len(not_blank_point), dtype=int)
                selected = np.concatenate( (not_blank_point, blank_point[out_selected]) )
            selected = np.sort(selected.squeeze())
            obj = obj[selected,:]

        
        if self.scale:
            scale_jitter = random.uniform(0.75, 1.33)
            obj[:,0] = scale_jitter * obj[:,0]
            scale_jitter = random.uniform(0.75, 1.33)
            obj[:,1] = scale_jitter * obj[:,1]
            scale_jitter = random.uniform(0.75, 1.33)
            obj[:,2] = scale_jitter * obj[:,2]
            jittered_data = np.random.normal(loc=0.0, scale=0.01, size=(obj.shape[0], 3)).clip(-0.01, 0.01)
            obj[:, 0:3] += jittered_data

        if self.rotate: 
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = np.asarray([[ np.cos(rotation_angle), np.sin(rotation_angle)],
                                             [-np.sin(rotation_angle), np.cos(rotation_angle)] ])
            xz = [0,2]
            obj[:, xz] = np.matmul(obj[:, xz], rotation_matrix)

        if self.erase>0:
            erase_ratio = np.random.random() * self.erase  # 0~0.875
            erase_length = round(erase_ratio * obj.shape[0])
            drop_start = np.random.randint(obj.shape[0] - erase_length)
            drop_end = min(drop_start+erase_length, obj.shape[0])
            obj[drop_start:drop_end, 3:] = 0.5 
        #if self.norm: 
        #    std = [0.1814, 0.4382, 0.1512, 0.2931, 0.3091, 0.3104]
        #    obj /=std 
        obj[:, 3:] -= 0.5 # - mean

        if self.channel == 3:
            return obj[:, 3:], target
        elif self.channel ==5:
            # [y, x^2+z^2, r, g, b]
            obj[:,2] = (obj[:,0]**2 + obj[:,2]**2)/10  
            return obj[:, 1:], target
            
        if self.D2:
            obj[:,2] = 0

        return obj, target

# Test Dataloader
if __name__ == '__main__':
    dst = Market3D('./2DMarket', flip=True, slim=0.3, scale=True, norm = True, erase=0.9, channel = 6)
    trainloader = data.DataLoader(dst.train(), batch_size=4)
    for i, data in enumerate(trainloader):
        objs, _, = data
        print(objs.shape)
        #print(torch.mean(objs,1))
        print(objs)
        break
