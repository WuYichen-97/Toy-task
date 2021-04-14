# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:12:31 2021

@author: coltonwu
"""

import torch
import torch.nn as nn
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('E:/Ours/PU-Net_pytorch-master')
import numpy as np

from pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
import pointnet2.pytorch_utils as pt_utils

def get_model(npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
    return PUNet(npoint, up_ratio, use_normal, use_bn, use_res)

class PUNet(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, use_normal=False, use_bn=False, use_res=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        self.npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]


                # PointnetSAModule(
                #     npoint=400,
                #     radius=0.05,
                #     nsample=32,
                #     mlp=[0] + mlps[0],
                #     use_xyz=True,
                #     use_res=False,
                #     bn=False)

        ## for 4 downsample layers
        in_ch = 0 if not use_normal else 2  #3
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
          if k==0:
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=False,
                    use_res=use_res,
                    bn=use_bn))
            in_ch = mlps[k][-1]
          else:
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=True,
                    use_res=use_res,
                    bn=use_bn))
            in_ch = mlps[k][-1]              

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1][-1], 64], 
                    bn=use_bn))
        
        ## feature Expansion
        in_ch = len(self.npoints) * 64 + 2 # 4 layers + input xyz   * 64 + 3
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        ## coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 2], activation=None, bn=False))  # [64, 3]


    def forward(self, points, npoint=None):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)

        ## points: bs, N, 3/6
        xyz = points[..., :2].contiguous()  # :3
        feats = points[..., 2:].transpose(1, 2).contiguous() \
            if self.use_normal else None
        feats = None
        print('points done!')
        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)
        print('downsample done!')
        ## upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            print(upk_feats)
            up_feats.append(upk_feats)
        print('upsample done!')    

        ## aggregation
        # [xyz, l0, l1, l2, l3]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1

        ## expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        ## reconstruction
        output = self.pcd_layer(r_feats) # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N

def sample_task(Batch, task_num):
    for i in range(Batch):
      task = []
      for j in range(task_num):
        a,b = np.random.uniform(0,2,2)
        task.append([a,b])
      if i == 0:
        task =  torch.tensor(task)
        out = task.unsqueeze(0)
      else:
        out = torch.cat((out,torch.tensor(task).unsqueeze(0)),0)
    return out
if __name__ == '__main__':
    model = PUNet(up_ratio=2, use_normal=True).cuda()
    points = torch.randn([1, 1024, 2]).float().cuda()
    output = model(points)
    #points = sample_task(5,400).float().cuda()
    #points = torch.rand([5, 400, 6]).float().cuda()
    #output = model(points, npoint=400)
    print('output_shape',output.shape)


    # npoints = [1024, 1024 // 2, 1024 // 4, 1024 // 8]
    # mlps = [
    #         [32, 32, 64],
    #         [64, 64, 128],
    #         [128, 128, 256],
    #         [256, 256, 512]
    #     ]
    # radius = [0.05, 0.1, 0.2, 0.3]
    # nsamples = [32, 32, 32, 32]

    # in_ch = 2
    # SA_modules = nn.ModuleList()
    # for k in range(len(npoints)):
    #   if k == 0:
    #     SA_modules.append(
    #       PointnetSAModule(
    #         npoint=npoints[k],
    #         radius=radius[k],
    #         nsample=nsamples[k],
    #         mlp=[in_ch] + mlps[k],
    #         use_xyz=False,
    #         use_res=False,
    #         bn=False)).cuda()
    #     in_ch = mlps[k][-1]
    #   else:
    #       SA_modules.append(
    #        PointnetSAModule(
    #          npoint=npoints[k],
    #          radius=radius[k],
    #          nsample=nsamples[k],
    #          mlp=[in_ch] + mlps[k],
    #          use_xyz=True,
    #          use_res=False,
    #          bn=False)).cuda()
    #       in_ch = mlps[k][-1]         

    # xyz = points[..., :2].contiguous()  # :3
    # #feats = points[..., 2:].transpose(1, 2).contiguous() 
    # feats = None
    # l_xyz, l_feats = [xyz], [feats]
    # for k in range(len(SA_modules)):
    #         print('k',k)
    #         lk_xyz, lk_feats = SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
    #         l_xyz.append(lk_xyz)
    #         l_feats.append(lk_feats)
    
    
    
    
    # feats = None
    # Net = PointnetSAModule(npoint=1024, radius=0.05, nsample=32, mlp=[2,32,32,64], use_xyz=True, use_res=False, bn=False).cuda()  
    # temp, feature = Net(xyz, feats, npoint = 1024)
    # print('temp.shape',temp.shape)
    # print('feature.shape',feature.shape)
    
    
    #Net(l_xyz[k], l_feats[k], npoint=npoints[k])    
    # points = torch.randn([1, 1024, 6]).float().cuda()
    # output = model(points)
    # print(output.shape)