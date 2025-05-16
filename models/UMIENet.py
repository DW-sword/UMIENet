import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import kornia

class CLAHE(nn.Module):
    def __init__(self, clipLimit=4.0, tileGridSize=(8, 8)):
        super(CLAHE, self).__init__()
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
    def forward(self,x):
        B, C, H, W = x.shape
        out = kornia.enhance.equalize_clahe(x/255,clip_limit=self.clipLimit,grid_size=self.tileGridSize)
        return out*255

class WB(nn.Module):
    def __init__(self):
        super(WB, self).__init__()
    def forward(self,img,wb_params):
        B, C, H, W = img.shape
        means = img.view(B, C, -1).mean(dim=2)  # Shape: [B, C]
        mean_intensity = means.mean(dim=1, keepdim=True)  # Average across channels, Shape: [B, 1]
        scale_factors = mean_intensity / means  # Shape: [B, C]
        scale_factors = scale_factors * wb_params
        balanced_img = img * scale_factors.view(B, C, 1, 1)
        balanced_img = torch.clamp(balanced_img, 0, 255)
        return balanced_img

class MSR(nn.Module):
    def __init__(self):
        super(MSR, self).__init__()

    def forward(self, img,sigmas):
        img = img.float() + 1e-6  # Convert to float and add a small constant
        B, C, H, W = img.shape
        retinex_list = []
        for b in range(B):
            img_b = img[b:b+1]
            retinex_b = []
            for sigma in sigmas[b]:
                kernel_size = (int(2 * sigma.item() + 1), int(2 * sigma.item() + 1))                                                                          
                illumination = kornia.filters.gaussian_blur2d(img_b,kernel_size,(sigma,sigma)) + 1e-6
                retinex_b.append((torch.log(img_b) - torch.log(illumination)).squeeze(0))
            retinex_b = torch.stack(retinex_b,0).sum(0)/sigmas.shape[1]
            retinex_list.append(retinex_b)
        retinex = torch.stack(retinex_list,0)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
        return retinex

class BB(nn.Module):    #Basic Block (BB)
    def __init__(self,channel):                                
        super(BB,self).__init__()

        self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)   
        self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)  
        self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

        self.attn = Sea_Attention(channel, key_dim=16, num_heads=1)

        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
    def forward(self,x):
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_out_1 = self.act(self.norm(self.conv_out(x_2)) + x)

        x_out_2 = self.attn(x)
        x_out = x_out_1 + x_out_2
        return	x_out

class Conv2d_BN(nn.Module):
    def __init__(self,in_channel,out_channel,ks=1,stride=1,pad=0,dilation=1,groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self,x):
        return self.bn(self.conv(x))

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x

class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                    attn_ratio=2,
                    activation=nn.ReLU,
                    norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads

        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1 )
        self.to_k = Conv2d_BN(dim, nh_kd, 1 )
        self.to_v = Conv2d_BN(dim, self.dh, 1 )
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim ))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh ))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh ))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        self.dwconv = Conv2d_BN(self.dh,  self.dh, ks=3, stride=1, pad=1, dilation=1,
                    groups=self.dh )
        self.act = activation()
        self.pwconv = Conv2d_BN(self.dh, dim, ks=1 )

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)


        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))


        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        return xx

class En_Decoder(nn.Module):
    def __init__(self,channel):
        super(En_Decoder,self).__init__()

        self.el = BB(channel)
        self.em = BB(channel*2)
        self.es = BB(channel*4)
        self.ds = BB(channel*4)
        self.dm = BB(channel*2)
        self.dl = BB(channel)

        self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
        self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

        self.conv_in = nn.Conv2d(12,channel,kernel_size=3,stride=1,padding=1,bias=False)

        self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


    def _upsample(self,x,y):
        _,_,H,W = y.size()
        return F.interpolate(x,size=(H,W),mode='bilinear')

    def forward(self,x1,x2,x3,x4):
             
        x_elin = torch.cat((x1,x2,x3,x4),1) + self.conv_in(x1+x2+x3+x4)

        elout = self.el(x_elin)
        emout = self.em(self.conv_eltem(self.maxpool(elout)))  
        esout = self.es(self.conv_emtes(self.maxpool(emout)))

        dsout = self.ds(esout)
        dmout = self.dm(self._upsample(self.conv_dstdm(dsout),emout) + emout)
        dlout = self.dl(self._upsample(self.conv_dmtdl(dmout),elout) + elout)

        x_out = self.conv_out(dlout)

        return x_out

class ParamsPredection(nn.Module):
    def __init__(self,num=6):
        super(ParamsPredection,self).__init__()
        self.conv = nn.Conv2d(12,24,kernel_size=3,stride=1,padding=1)
        self.act = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((8,8))
        self.fc = nn.Linear(1536,num)
        self.act_2 = nn.Sigmoid()
    def forward(self,x):
        x = self.avgpool(self.act(self.conv(x)))
        x = x.view(x.size(0),-1)
        params = self.fc(x)

        # 参数限制范围
        rgb_wights = self.act_2(params[:, 0:3]) * 2 + 0.5
        sigmas = torch.round(self.act_2(params[:, 3:]) * 100 + 1)
        
        return rgb_wights,sigmas  # [bs,num]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.clahe = CLAHE()  # CLAHE算法
        self.wb = WB()  # WB算法
        self.msr = MSR()  # MSR算法
        self.conv_im = nn.Conv2d(3,12,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)  # 图像特征提取卷积层
        self.conv_clahe = nn.Conv2d(3,12,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)  # CLAHE特征提取卷积层
        self.conv_wb = nn.Conv2d(3,12,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
        self.conv_msr = nn.Conv2d(3,12,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
        self.BB_im = BB(12)
        self.BB_clahe = BB(12)
        self.BB_wb = BB(12)
        self.BB_msr = BB(12)
        self.params_pred = ParamsPredection(num=6)

        self.ED = En_Decoder(12*4)

        self.conv_domian = nn.Conv2d(48, 256, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化层
 
    def forward(self,img):
        # feature extraction特征提取模块
        img_feat = torch.clamp(img/255,1e-10,1.0)
        img_feat = self.BB_im(self.conv_im(img_feat))

        wb_params,msr_params = self.params_pred(img_feat)
        # print(wb_params,msr_params)

        clahe_feat = self.clahe(img)
        clahe_feat = torch.clamp(clahe_feat/255,1e-10,1.0)
        clahe_feat = self.BB_clahe(self.conv_clahe(clahe_feat))

        wb_feat = self.wb(img,wb_params)
        wb_feat = torch.clamp(wb_feat/255,1e-10,1.0)
        wb_feat = self.BB_wb(self.conv_wb(wb_feat))

        msr_feat = self.msr(img,msr_params)
        msr_feat = torch.clamp(msr_feat/255,1e-10,1.0)
        msr_feat = self.BB_msr(self.conv_msr(msr_feat))


        # encoder-decoder based特征融合模块
        output = self.ED(img_feat,clahe_feat,wb_feat,msr_feat)

        return output  # [bs,3,h,w]

    def extract_features(self,img):
        # feature extraction特征提取模块
        img_feat = torch.clamp(img/255,1e-10,1.0)
        img_feat = self.BB_im(self.conv_im(img_feat))

        wb_params,msr_params = self.params_pred(img_feat)

        clahe_feat = self.clahe(img)
        clahe_feat = torch.clamp(clahe_feat/255,1e-10,1.0)
        clahe_feat = self.BB_clahe(self.conv_clahe(clahe_feat))

        wb_feat = self.wb(img,wb_params)
        wb_feat = torch.clamp(wb_feat/255,1e-10,1.0)
        wb_feat = self.BB_wb(self.conv_wb(wb_feat))

        msr_feat = self.msr(img,msr_params)
        msr_feat = torch.clamp(msr_feat/255,1e-10,1.0)
        msr_feat = self.BB_msr(self.conv_msr(msr_feat))

        features = torch.cat([img_feat, clahe_feat, wb_feat, msr_feat], dim=1) #[bs,48,h,w]
        features = self.conv_domian(features)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)

        return features

