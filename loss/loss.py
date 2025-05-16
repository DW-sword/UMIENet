import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:31]
        self.layer_weights = [1/32, 1/16, 1/8, 1/4, 1]
        self.selected_layers = [2, 7, 12, 21, 30]  # Corresponding to the layers after which we extract features

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def extract_features(self,x):
            features = []
            for i, layer in self.vgg._modules.items():
                layer.to(x.device)
                x = layer(x)
                if int(i) in self.selected_layers:
                    features.append(x)
            return features
    
    # I_r: 推理图像 I_g: 真值图像 I_d: 原始退化图像
    def forward(self, I_r, I_g, I_d):
        # L1-Norm Loss
        L1_loss = torch.mean(torch.abs(I_r - I_g))

        # Color Loss
        I_r_flat = I_r.view(I_r.size(0), -1)
        I_g_flat = I_g.view(I_g.size(0), -1)
        color_loss = 1 - torch.sum(I_r_flat * I_g_flat, dim=1) / (torch.norm(I_r_flat, dim=1) * torch.norm(I_g_flat, dim=1))
        color_loss = torch.mean(color_loss)

        # Contrastive Regularization Loss

        I_r_features = self.extract_features(I_r)
        I_g_features = self.extract_features(I_g)
        I_d_features = self.extract_features(I_d)

        cr_loss = 0
        for i in range(len(self.selected_layers)):
            psi_r = I_r_features[i]
            psi_g = I_g_features[i]
            psi_d = I_d_features[i]

            cr_loss += self.layer_weights[i] * (
                torch.mean(torch.abs(psi_r - psi_g)) / torch.mean(torch.abs(psi_d - psi_g))
            )

        total_loss = L1_loss + color_loss + cr_loss
        return total_loss

if __name__ == '__main__':
    loss = Loss()
    I_r, I_g, I_d = torch.randn(4, 3, 480, 640), torch.randn(4, 3, 480, 640), torch.randn(1, 3, 480, 640)
    total_loss = loss(I_r, I_g, I_d)
    print(total_loss.item())

