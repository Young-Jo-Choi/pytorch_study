import numpy as np
import cv2
import copy
import torch
from torch import nn


class GradCamModel(nn.Module):
    def __init__(self, conv:nn.Sequential, classifier:nn.Sequential):
        super().__init__()
        # self.model = copy.deepcopy(model)
        # self.conv = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
        #              self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4)
        # self.avg = self.model.avgpool
        # self.fc = self.model.fc
        self.conv = conv
        self.classifier = classifier # 수정


        # placeholder for gradients
        self.gradient = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradient = grad

    def forward(self,x):

        x = self.conv(x)
        self.conv_result = x

        # register the hook
        # https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
        # Tensor.register_hook(hook) -> Tensor or None
        h = x.register_hook(self.activations_hook)

        # apply remaining layers
        # x = self.avg(x)
        # # flatten
        # x = x.view((1, -1))
        # x = self.fc(x)
        x = self.classifier(x)
        self.prediction = x.argmax()
        self.logits = x
        return x

    def get_heatmap(self, x):
        '''after forward'''
        logits = self.logits
        prediction = self.prediction
        logits[:, prediction].backward()
        gradients = self.gradient
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.conv_result.detach()
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)

        self.gradient = None
        return heatmap