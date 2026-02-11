import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from math import sqrt
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, bn=False,
                 relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu is not None:
            y = self.relu(y)
        return y



def upsample(x, scale_factor=2, mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)



class DGModel_base(nn.Module):
    def __init__(self, pretrained=True, den_dropout1=0.3,den_dropout2=0.3, den_dropout3=0.3):
        super().__init__()

        self.den_dropout1=den_dropout1
        self.den_dropout2=den_dropout2
        self.den_dropout3=den_dropout3

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)

        self.enc1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.enc2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.enc3 = nn.Sequential(*list(vgg.features.children())[33:43])


        self.dec3 = nn.Sequential(
            ConvBlock(512, 1024, bn=True),
            ConvBlock(1024, 512, bn=True),
            ConvBlock(512, 256,kernel_size=1, padding=0, bn=True),
        )

        self.dec2 = nn.Sequential(
            ConvBlock(512, 512, bn=True),
            ConvBlock(512, 256, bn=True), 
            ConvBlock(256, 256,kernel_size=1, padding=0, bn=True),
        )

        self.dec1 = nn.Sequential(
            ConvBlock(256, 128, bn=True),
            ConvBlock(128, 128, bn=True),
            ConvBlock(128, 256,kernel_size=1, padding=0, bn=True),
        )
        
        
        self.den_dec1 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=1, padding=0,bn=True),
            nn.Dropout2d(p=den_dropout1)
        )
        
        self.den_dec2 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=1, padding=0,bn=True),
            nn.Dropout2d(p=den_dropout2)
        )
        self.den_dec3 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=1, padding=0,bn=True),
            nn.Dropout2d(p=den_dropout3)
        )


        self.den_head1 = nn.Sequential(
            ConvBlock(128, 1, kernel_size=1, padding=0),
        )
        self.den_head2 = nn.Sequential(
            ConvBlock(128, 1, kernel_size=1, padding=0),
        )
        self.den_head3 = nn.Sequential(
            ConvBlock(128, 1, kernel_size=1, padding=0),
        )


    def jsd(self, logits1, logits2):
        b, c, h, w = logits1.shape
        feature_layer1_flat = logits1.view(b, c, -1)  # [b, 128, 6400]
        feature_layer2_flat = logits2.view(b, c, -1)  # [b, 128, 6400]
        p1 = F.softmax(feature_layer1_flat, dim=1)
        p2 = F.softmax(feature_layer2_flat, dim=1)
        jsd = F.mse_loss(p1, p2)
        
        return -jsd



    def forward_similarity(self, y1, y2, y3):

        b, c, h, w = y1.shape
          
        logits_12 = torch.sum(y1 * y2, dim=1, keepdim=True)/sqrt(c)   # [b, 1, h, w]
        logits_13 = torch.sum(y1 * y3, dim=1, keepdim=True)/sqrt(c)   # [b, 1, h, w]
        logits_23 = torch.sum(y2 * y3, dim=1, keepdim=True)/sqrt(c)   # [b, 1, h, w]

        score1 = (logits_12 + logits_13) / 2
        score2 = (logits_12 + logits_23) / 2
        score3 = (logits_13 + logits_23) / 2

        scores = torch.cat([score1, score2, score3], dim=1)  # [b, 3, 80, 80]

        out_weights = F.softmax(-scores, dim=1)  # [b, 3, 80, 80]

        inner_weights1 = F.softmax(-(y1 * y2 + y1 * y3), dim=1)  # b 128 80 80
        inner_weights2 = F.softmax(-(y2 * y3 + y2 * y1), dim=1)
        inner_weights3 = F.softmax(-(y1 * y3 + y3 * y2), dim=1)

        #   b 256 6400 *b  6400 6400
        y1_weighted = y1 * out_weights[:, 0:1, :, :] * inner_weights1  # [b, c, h, w]
        y2_weighted = y2 * out_weights[:, 1:2, :, :] * inner_weights2  # [b, c, h, w]
        y3_weighted = y3 * out_weights[:, 2:3, :, :] * inner_weights3  # [b, c, h, w]

        
        return y1_weighted, y2_weighted, y3_weighted,self.jsd(y1_weighted,y2_weighted)+self.jsd(y1_weighted,y3_weighted)+self.jsd(y2_weighted, y3_weighted)

    def ronghemidut(self, y1, y2, y3):

        y_cat_origin = torch.cat([y1, y2, y3], dim=1)  # [b, 3, h, w]

        y_max = y_cat_origin.max(dim=1, keepdim=True)[0]

        y_sum = y_cat_origin.sum(dim=1, keepdim=True) + 1e-6
        y_conf = y_cat_origin / y_sum


        y_weighted = y_conf * y_cat_origin

        y1_weighted = y_weighted[:, 0:1, :, :]
        y2_weighted = y_weighted[:, 1:2, :, :]
        y3_weighted = y_weighted[:, 2:3, :, :]
        
        y_weighted_mean = y_weighted.mean(dim=1, keepdim=True)  # [b, 1, h, w]

        return y_max, y_weighted_mean, y1_weighted, y2_weighted, y3_weighted,y1,y2,y3
    


    def forward_fe(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        y3 = self.dec3(x3)
        y3 = upsample(y3, scale_factor=4)

        y2 = self.dec2(x2)
        y2 = upsample(y2, scale_factor=2)
        
        y1 = self.dec1(x1)

        return y1, y2, y3, x1, x2, x3
        

    
    def compute_attention_weights(self, feature_map):

        b, c, h, w = feature_map.shape

        avg_pooled = feature_map.mean(dim=(2, 3), keepdim=True)  # [b, c, 1, 1]

        max_pooled, _ = torch.max(feature_map, dim=2, keepdim=True)  # [b, c, 1, w]
        max_pooled, _ = torch.max(max_pooled, dim=3, keepdim=True)  # [b, c, 1, 1]

        thresholds = (avg_pooled + max_pooled) / 2  # [b, c, 1, 1]

        weights = feature_map / (thresholds + 1e-8)
        new = feature_map * weights
        return new
        

    def forward(self, x):
        y1, y2, y3, x11, x21, x31 = self.forward_fe(x)

        y1 = self.compute_attention_weights(y1)
        y2 = self.compute_attention_weights(y2)
        y3 = self.compute_attention_weights(y3)

        y1, y2, y3,h = self.forward_similarity(y1, y2, y3)

        y_den1 = self.den_dec1(y1)
        y_den2 = self.den_dec2(y2)
        y_den3 = self.den_dec3(y3)

        y_den1_new, logit1 = self.forward_mem(y_den1, self.mem1)
        y_den2_new, logit2 = self.forward_mem(y_den2, self.mem2)
        y_den3_new, logit3 = self.forward_mem(y_den3, self.mem3)
       
        log1=self.mse(logit1,logit2)+self.mse(logit1,logit3)
        log2=self.mse(logit2,logit1)+self.mse(logit2,logit3)
        log3=self.mse(logit1,logit3)+self.mse(logit3,logit1)
        log=(log1+log2+log3)/2
        
        d1 = self.den_head1(y_den1_new)
        d2 = self.den_head2(y_den2_new)
        d3 = self.den_head3(y_den3_new)

        d, d_mean, y1_weighted, y2_weighted, y3_weighted,d1,d2,d3 = self.ronghemidut(d1, d2, d3)

        d = upsample(d, scale_factor=4)
        
        d1 = upsample(d1, scale_factor=4)
        d2 = upsample(d2, scale_factor=4)
        d3 = upsample(d3, scale_factor=4)


        return d,h, d_mean, y1_weighted, y2_weighted, y3_weighted,d1,d2,d3,log






class DGModel_mem(DGModel_base):
    def __init__(self, pretrained=True, den_dropout1=0.3,den_dropout2=0.3, den_dropout3=0.3, mem_dim1=128, mem_dim2=128,mem_dim3=128,mem_size1=1024,mem_size2=512,mem_size3=256):
        super().__init__(pretrained,den_dropout1, den_dropout2, den_dropout3)

        self.mem_size1 = mem_size1
        self.mem_size2 = mem_size2
        self.mem_size3 = mem_size3
        self.mem_dim1 = mem_dim1
        self.mem_dim2 = mem_dim2
        self.mem_dim3 = mem_dim3

        
        self.mem1 = nn.Parameter(torch.FloatTensor(1, self.mem_dim1, self.mem_size1).normal_(0.0, 1.0))  # 对应 y1
        self.mem2 = nn.Parameter(torch.FloatTensor(1, self.mem_dim2, self.mem_size2).normal_(0.0, 1.0))  # 对应 y2
        self.mem3 = nn.Parameter(torch.FloatTensor(1, self.mem_dim3, self.mem_size3).normal_(0.0, 1.0))  # 对应 y3


        self.den_dec1 = nn.Sequential(
            ConvBlock(256, self.mem_dim1, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout1)
        )
        self.den_dec2 = nn.Sequential(
            ConvBlock(256, self.mem_dim2, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout2)
        )
        self.den_dec3 = nn.Sequential(
            ConvBlock(256, self.mem_dim3, kernel_size=1, padding=0, bn=True),
            nn.Dropout2d(p=den_dropout3)
        )

        self.den_head1 = nn.Sequential(
            ConvBlock(self.mem_dim1, 1, kernel_size=1, padding=0)
        )
        self.den_head2 = nn.Sequential(
            ConvBlock(self.mem_dim2, 1, kernel_size=1, padding=0)
        )
        self.den_head3 = nn.Sequential(
            ConvBlock(self.mem_dim3, 1, kernel_size=1, padding=0),
        )

    def forward_mem(self, y, M):

        b, k, h, w = y.shape
        m = M.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.view(b, k, -1)

        logits = torch.bmm(m_key, y_) / sqrt(k)

        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.view(b, k, h, w)

        return y_new_, logits

    def forward(self, x):
        y1, y2, y3, _, _, _ = self.forward_fe(x)
        y_den1, y_den2, y_den3 = self.forward_similarity(y1, y2, y3)

        y_den1 = self.den_dec1(y_den1)
        y_den2 = self.den_dec2(y_den2)
        y_den3 = self.den_dec3(y_den3)

        y_den1_new, _ = self.forward_mem(y_den1, self.mem1)
        d1 = self.den_head1(y_den1_new)


        y_den2_new, _ = self.forward_mem(y_den2, self.mem2)
        d2 = self.den_head2(y_den2_new)


        y_den3_new, _ = self.forward_mem(y_den3, self.mem3)
        d3 = self.den_head3(y_den3_new)

        d, d_mean, y1_weighted, y2_weighted, y3_weighted = self.ronghemidut(d1, d2, d3)

        d = upsample(d, scale_factor=4)


        return d, d_mean, y1_weighted, y2_weighted, y3_weighted


class DGModel_memadd(DGModel_mem):
    def __init__(self, pretrained=True, den_dropout1=0.3, den_dropout2=0.3, den_dropout3=0.3, mem_dim1=128, mem_dim2=128, mem_dim3=128,mem_size=512,err_thrs1=0.5, err_thrs2=0.5, err_thrs3=0.5,):
        super().__init__(pretrained, den_dropout1, den_dropout2, den_dropout3, mem_dim1, mem_dim2, mem_dim3,mem_size)
        self.mem_size = mem_size

        self.mem_dim1 = mem_dim1
        self.mem_dim2 = mem_dim2
        self.mem_dim3 = mem_dim3

        self.err_thrs1 = err_thrs1
        self.err_thrs2 = err_thrs2
        self.err_thrs3 = err_thrs3

       
    def jsd1(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2):

        y11, y21, y31, x11, x21, x31 = self.forward_fe(img1)
        y12, y22, y32, x12, x22, x32 = self.forward_fe(img2)


        y1_in1 = F.instance_norm(y11, eps=1e-5)
        y1_in2 = F.instance_norm(y12, eps=1e-5)

        y2_in1 = F.instance_norm(y21, eps=1e-5)
        y2_in2 = F.instance_norm(y22, eps=1e-5)

        y3_in1 = F.instance_norm(y31, eps=1e-5)
        y3_in2 = F.instance_norm(y32, eps=1e-5)


        e_y1 = torch.abs(y1_in1 - y1_in2)
        e_mask1 = (e_y1 < self.err_thrs1).clone().detach()
        y1_den_masked1 = y11 * e_mask1
        y1_den_masked2 = y12 * e_mask1

        e_y2 = torch.abs(y2_in1 - y2_in2)
        e_mask2 = (e_y2 < self.err_thrs2).clone().detach()
        y2_den_masked1 = y21 * e_mask2
        y2_den_masked2 = y22 * e_mask2

        e_y3 = torch.abs(y3_in1 - y3_in2)
        e_mask3 = (e_y3 < self.err_thrs3).clone().detach()
        y3_den_masked1 = y31 * e_mask3
        y3_den_masked2 = y32 * e_mask3

        y1_den_new1,  y2_den_new1,  y3_den_new1 = self.forward_similarity(y1_den_masked1, y2_den_masked1, y3_den_masked1)
        y1_den_new2,  y2_den_new2,  y3_den_new2 = self.forward_similarity(y1_den_masked2, y2_den_masked2, y3_den_masked2)

        y1_den_new1 = self.den_dec1(y1_den_new1)
        y1_den_new2 = self.den_dec1(y1_den_new2)

        y2_den_new1 = self.den_dec2( y2_den_new1)
        y2_den_new2 = self.den_dec2( y2_den_new2)

        y3_den_new1= self.den_dec3(y3_den_new1)
        y3_den_new2 = self.den_dec3(y3_den_new2)

        y1_den_new1, logits1_1 = self.forward_mem(y1_den_new1, self.mem1)
        y1_den_new2, logits1_2 = self.forward_mem(y1_den_new2, self.mem1)

        y2_den_new1, logits2_1 = self.forward_mem(y2_den_new1, self.mem2)
        y2_den_new2, logits2_2 = self.forward_mem(y2_den_new2, self.mem2)

        y3_den_new1, logits3_1 = self.forward_mem(y3_den_new1, self.mem3)
        y3_den_new2, logits3_2 = self.forward_mem(y3_den_new2, self.mem3)

        loss_con1 = self.jsd(logits1_1, logits1_2)
        loss_con2 = self.jsd(logits2_1, logits2_2)
        loss_con3 = self.jsd(logits3_1, logits3_2)

        loss_con = loss_con1 + loss_con2 + loss_con3

        d1_1 = self.den_head1(y1_den_new1)
        d1_2 = self.den_head1(y1_den_new2)

        d2_1 = self.den_head2(y2_den_new1)
        d2_2 = self.den_head2(y2_den_new2)

        d3_1 = self.den_head3(y3_den_new1)
        d3_2 = self.den_head3(y3_den_new2)

        
        d1, d_mean1, y11_weighted, y21_weighted, y31_weighted = self.ronghemidut(d1_1, d2_1, d3_1)
        d2, d_mean2, y12_weighted, y22_weighted, y32_weighted = self.ronghemidut(d1_2, d2_2, d3_2)

        d1 = upsample(d1, scale_factor=4)
        d2 = upsample(d2, scale_factor=4)

        y11_weighted = upsample(y11_weighted, scale_factor=4)
        y21_weighted = upsample(y21_weighted, scale_factor=4)
        y31_weighted = upsample(y31_weighted, scale_factor=4)
        y12_weighted = upsample(y12_weighted, scale_factor=4)
        y22_weighted = upsample(y22_weighted, scale_factor=4)
        y32_weighted = upsample(y32_weighted, scale_factor=4)

        d_mean1 = upsample(d_mean1, scale_factor=4)
        d_mean2 = upsample(d_mean2, scale_factor=4)
        return d1, d2, loss_con, (y11_weighted, y21_weighted, y31_weighted), (y12_weighted, y22_weighted, y32_weighted), (d_mean1, d_mean2)













class DGModel_cls(DGModel_base):
    def __init__(self, pretrained=True, den_dropout1=0.3, den_dropout2=0.3, den_dropout3=0.3, cls_dropout1=0.3,cls_dropout2=0.3,cls_dropout3=0.3,cls_thrs1=0.5,cls_thrs2=0.5,cls_thrs3=0.5):
        super().__init__(pretrained, den_dropout1, den_dropout2, den_dropout3)

        self.cls_dropout1 = cls_dropout1
        self.cls_dropout2 = cls_dropout2
        self.cls_dropout3 = cls_dropout3
    
        self.cls_thrs1 = cls_thrs1
        self.cls_thrs2 = cls_thrs2
        self.cls_thrs3 = cls_thrs3


        self.cls_head3 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout3),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )

        

    def transform_cls_map_gt(self, c_gt,scale_factor=4):
        return upsample(c_gt, scale_factor=scale_factor, mode='nearest')

    def transform_cls_map_pred(self, c, cls_thrs, scale_factor=4):
        c_new = c.clone().detach()
        c_new[c < cls_thrs] = 0
        c_new[c >= cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=scale_factor, mode='nearest')
       
        return c_resized

    def transform_cls_map(self, c, c_gt=None, scale_factor=4, cls_thrs=0.5):
        if c_gt is not None:
            return self. transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c, cls_thrs=self.cls_thrs3, scale_factor=scale_factor)


    
    def forward(self, x, c_gt=None):
        y1, y2, y3, x11, x21, x31 = self.forward_fe(x)
       
        
        c = self.cls_head3(x31)
        c_resized = self.transform_cls_map(c, c_gt)
        
        y1 = self.compute_attention_weights(y1)
        y2 = self.compute_attention_weights(y2)
        y3 = self.compute_attention_weights(y3)

        y1, y2, y3,h = self.forward_similarity(y1, y2, y3)
        
        y_den1 = self.den_dec1(y1)
        y_den2 = self.den_dec2(y2)
        y_den3 = self.den_dec3(y3)


        d1 = self.den_head1(y_den1)
        d2 = self.den_head2(y_den2)
        d3 = self.den_head3(y_den3)

        d, d_mean, y1_weighted, y2_weighted, y3_weighted = self.ronghemidut(d1, d2, d3)
        d = upsample(d*c_resized, scale_factor=4)

        y1_weighted = upsample(y1_weighted*c_resized, scale_factor=4)
        y2_weighted = upsample(y2_weighted*c_resized, scale_factor=4)
        y3_weighted = upsample(y3_weighted*c_resized, scale_factor=4)

        d_mean = upsample(d_mean*c_resized, scale_factor=4)

        return d,h, d_mean, y1_weighted, y2_weighted, y3_weighted,c







class DGModel_memcls(DGModel_mem):
    def __init__(self, pretrained=True,den_dropout1=0.3, den_dropout2=0.3, den_dropout3=0.3,  mem_dim1=128, mem_dim2=128, mem_dim3=128,mem_size1=1024,mem_size2=512,mem_size3=256,cls_dropout1=0.3,
                 cls_dropout2=0.3, cls_dropout3=0.3, cls_thrs3=0.5, cls_thrs2=0.5, cls_thrs1=0.5):
        super().__init__(pretrained,  den_dropout1, den_dropout2, den_dropout3, mem_dim1, mem_dim2, mem_dim3,mem_size1,mem_size2,mem_size3)

        self.cls_dropout1 = cls_dropout1
        self.cls_dropout2 = cls_dropout2
        self.cls_dropout3 = cls_dropout3

        self.cls_thrs1 = cls_thrs1
        self.cls_thrs2 = cls_thrs2
        self.cls_thrs3 = cls_thrs3

        self.cls_head3 = nn.Sequential(
            ConvBlock(512, 256, bn=True),
            nn.Dropout2d(p=self.cls_dropout3),
            ConvBlock(256, 1, kernel_size=1, padding=0, relu=False),
            nn.Sigmoid()
        )



    def transform_cls_map_gt(self, c_gt,scale_factor=4):
        return upsample(c_gt, scale_factor=scale_factor, mode='nearest')

    def transform_cls_map_pred(self, c, cls_thrs, scale_factor=4):
        c_new = c.clone().detach()
        c_new[c < cls_thrs] = 0
        c_new[c >= cls_thrs] = 1
        c_resized = upsample(c_new, scale_factor=scale_factor, mode='nearest')
       
        return c_resized

    def transform_cls_map(self, c, c_gt=None, scale_factor=4, cls_thrs=0.5):
        if c_gt is not None:
            return self. transform_cls_map_gt(c_gt)
        else:
            return self.transform_cls_map_pred(c, cls_thrs=self.cls_thrs3, scale_factor=scale_factor)


    
    def forward(self, x, c_gt=None):
        y1, y2, y3, x11, x21, x31 = self.forward_fe(x)

        c = self.cls_head3(x31)
        c_resized = self.transform_cls_map(c, c_gt)
        
        y1 = self.compute_attention_weights(y1)
        y2 = self.compute_attention_weights(y2)
        y3 = self.compute_attention_weights(y3)

        y1, y2, y3, h = self.forward_similarity(y1, y2, y3)

        y1_den1 = self.den_dec1(y1)
        y1_den2 = self.den_dec2(y2)
        y1_den3 = self.den_dec3(y3)


        y1_den1, _ = self.forward_mem(y1_den1, self.mem1)
        y1_den2, _ = self.forward_mem(y1_den2, self.mem2)
        y1_den3, _ = self.forward_mem(y1_den3, self.mem3)
        
        d1 = self.den_head1(y1_den1)
        d2 = self.den_head2(y1_den2)
        d3 = self.den_head3(y1_den3)
        
        d, d_mean, y1_weighted, y2_weighted, y3_weighted,_,_,_ = self.ronghemidut(d1, d2, d3)
        d = upsample(d * c_resized, scale_factor=4)

        return d, h, d_mean, y1_weighted, y2_weighted, y3_weighted, c





class DGModel_final(DGModel_memcls):
    def __init__(self, pretrained=True, den_dropout1=0.3, den_dropout2=0.3, den_dropout3=0.3, mem_dim1=128, 
                 mem_dim2=128, mem_dim3=128, mem_size1=1024,mem_size2=512,mem_size3=256, cls_dropout1=0.3,cls_dropout2=0.3, cls_dropout3=0.3,cls_thrs1=0.5, cls_thrs2=0.5, cls_thrs3=0.5
                 ,err_thrs3=0.5, err_thrs2=0.5, err_thrs1=0.5,has_err_loss=False):
        super().__init__(pretrained, den_dropout1, den_dropout2, den_dropout3, mem_dim1, mem_dim2, mem_dim3, mem_size1,mem_size2,mem_size3,cls_thrs1, cls_thrs2, cls_thrs3,cls_dropout1,cls_dropout2,cls_dropout3)

        self.err_thrs1 = err_thrs1
        self.err_thrs2 = err_thrs2
        self.err_thrs3 = err_thrs3
        self.has_err_loss = has_err_loss


    def jsd1(self, logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        jsd = F.mse_loss(p1, p2)
        return jsd

    def forward_train(self, img1, img2, c_gt=None):
        y11, y21, y31, x11, x21, x31 = self.forward_fe(img1)
        y12, y22, y32, x12, x22, x32 = self.forward_fe(img2)

        y1_in1 = F.instance_norm(y11, eps=1e-5)
        y1_in2 = F.instance_norm(y12, eps=1e-5)

        y2_in1 = F.instance_norm(y21, eps=1e-5)
        y2_in2 = F.instance_norm(y22, eps=1e-5)

        y3_in1 = F.instance_norm(y31, eps=1e-5)
        y3_in2 = F.instance_norm(y32, eps=1e-5)

        e_y1 = torch.abs(y1_in1 - y1_in2)
        e_mask1 = (e_y1 < self.err_thrs1).clone().detach()
        e_y2 = torch.abs(y2_in1 - y2_in2)
        e_mask2 = (e_y2 < self.err_thrs2).clone().detach()
        e_y3 = torch.abs(y3_in1 - y3_in2)
        e_mask3 = (e_y3 < self.err_thrs3).clone().detach()

        
        y_den_new1_1 = self.compute_attention_weights(y11)
        y_den_new2_1 = self.compute_attention_weights(y21)
        y_den_new3_1 = self.compute_attention_weights(y31)
        
        y_den_new1_2 = self.compute_attention_weights(y12)
        y_den_new2_2 = self.compute_attention_weights(y22)
        y_den_new3_2 = self.compute_attention_weights(y32)

        
        y11, y21, y31,h1 = self.forward_similarity(y_den_new1_1, y_den_new2_1, y_den_new3_1)
        y12, y22, y32,h2 = self.forward_similarity(y_den_new1_2, y_den_new2_2, y_den_new3_2)


        y1_den_masked1 = self.den_dec1(y11* e_mask1)
        y2_den_masked1 = self.den_dec2(y21* e_mask2)
        y3_den_masked1 = self.den_dec3(y31* e_mask3)

        y1_den_masked2 = self.den_dec1(y12* e_mask1)
        y2_den_masked2 = self.den_dec2(y22* e_mask2)
        y3_den_masked2 = self.den_dec3(y32* e_mask3)
        
        y_den_new1_1, logits1_1 = self.forward_mem(y1_den_masked1, self.mem1)
        y_den_new2_1, logits2_1 = self.forward_mem(y2_den_masked1, self.mem2)
        y_den_new3_1, logits3_1 = self.forward_mem(y3_den_masked1, self.mem3)

        y_den_new1_2, logits1_2 = self.forward_mem(y1_den_masked2, self.mem1)
        y_den_new2_2, logits2_2 = self.forward_mem(y2_den_masked2, self.mem2)
        y_den_new3_2, logits3_2 = self.forward_mem(y3_den_masked2, self.mem3)
        
        loss_con = (self.jsd1(logits1_1, logits1_2) + self.jsd1(logits2_1, logits2_2) + self.jsd1(logits3_1, logits3_2))
        

        d11 = self.den_head1(y_den_new1_1)
        d21 = self.den_head2(y_den_new2_1)
        d31 = self.den_head3(y_den_new3_1)

        d12 = self.den_head1(y_den_new1_2)
        d22 = self.den_head2(y_den_new2_2)
        d32 = self.den_head3(y_den_new3_2)
   

        c1 = self.cls_head3(x31)
        c2 = self.cls_head3(x32)
        c_resized_gt = self.transform_cls_map_gt(c_gt)
        c_resized1 = self.transform_cls_map_pred(c1, self.cls_thrs3)
        c_resized2 = self.transform_cls_map_pred(c2, self.cls_thrs3)
        c_err = torch.abs(c_resized1 - c_resized2)
        c_resized = torch.clamp(c_resized_gt + c_err, 0, 1)
        

        dx1,d_mean1,y11_mask, y21_mask, y31_mask,y11_weighted, y21_weighted, y31_weighted = self.ronghemidut(d11, d21, d31)
        dx2,d_mean2,y12_mask, y22_mask, y32_mask,y12_weighted, y22_weighted, y32_weighted = self.ronghemidut(d12, d22, d32)

        dx1 = upsample(dx1* c_resized, scale_factor=4) 
        dx2 = upsample(dx2* c_resized, scale_factor=4)
        
        y11_weighted = upsample(y11_weighted*c_resized, scale_factor=4)
        y21_weighted = upsample(y21_weighted*c_resized, scale_factor=4)
        y31_weighted = upsample(y31_weighted*c_resized, scale_factor=4)

        y12_weighted = upsample(y12_weighted*c_resized, scale_factor=4)
        y22_weighted = upsample(y22_weighted*c_resized, scale_factor=4)
        y32_weighted = upsample(y32_weighted*c_resized, scale_factor=4)
        
        d_mean1 = upsample(d_mean1, scale_factor=4)
        d_mean2 = upsample(d_mean2, scale_factor=4)

        y11_mask = upsample(y11_mask, scale_factor=4)
        y21_mask = upsample(y21_mask, scale_factor=4)
        y31_mask = upsample(y31_mask, scale_factor=4)
        y12_mask = upsample(y12_mask, scale_factor=4)
        y22_mask = upsample(y22_mask, scale_factor=4)
        y32_mask = upsample(y32_mask, scale_factor=4)
        
        return dx1, dx2, (d_mean1,y11_mask, y21_mask, y31_mask,y11_weighted, y21_weighted, y31_weighted), (d_mean2,y12_mask, y22_mask, y32_mask,y12_weighted, y22_weighted, y32_weighted), (c1, c2), loss_con, 1,h1,h2
