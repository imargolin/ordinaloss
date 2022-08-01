# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:49:54 2022

@author: imargolin
"""

import numpy as np
import torch

from torch import nn
from torchvision import models

print(f"loaded {__name__}")

def classification_model_resnet(architecture:str, 
                                num_classes:int) -> nn.Module:
    
    
    
    all_architectures = {"resnet18": models.resnet18, 
                         "resnet34": models.resnet34,
                         "resnet50": models.resnet50, 
                         "resnet101": models.resnet101,
                         "resnet152": models.resnet152}
    
    all_weights = {"resnet18":models.ResNet18_Weights.DEFAULT,
                   "resnet34":models.ResNet34_Weights.DEFAULT,
                   "resnet50":models.ResNet50_Weights.DEFAULT,
                   "resnet101":models.ResNet101_Weights.DEFAULT,
                   "resnet152":models.ResNet152_Weights.DEFAULT}
    
    assert architecture in all_architectures, f"Should be one of {all_architectures.keys()}"
    
    model = all_architectures[architecture](weights=all_weights[architecture], 
                                               progress = True)
    
    
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)    
    
    return model

def classification_model_vgg(architecture:str, 
                             num_classes:int) -> nn.Module:
    
    all_architectures = {"vgg16": models.vgg16, 
                         "vgg19": models.vgg19,
                         "vgg16_bn": models.vgg16_bn, 
                         "vgg19_bn": models.vgg19_bn}
    
    all_weights = {"vgg16":    models.vgg.VGG16_Weights.DEFAULT,
                   "vgg19":    models.vgg.VGG19_Weights.DEFAULT,
                   "vgg16_bn": models.vgg.VGG16_BN_Weights.DEFAULT,
                   "vgg19_bn": models.vgg.VGG19_BN_Weights.DEFAULT}
    
    assert architecture in all_architectures, f"Should be one of {all_architectures.keys()}"
    
    model = all_architectures[architecture](weights=all_weights[architecture], 
                                           progress = True)
    
    
    num_ftrs = model.classifier[6].in_features
    feature_model = list(model.classifier.children())
    feature_model.pop()
    feature_model.append(nn.Linear(num_ftrs, num_classes))
    model.classifier = nn.Sequential(*feature_model)
    
    return model


def classification_model_densenet(architecture:str, 
                             num_classes:int) -> nn.Module:
    
    all_architectures = {"densenet121": models.densenet121, 
                         "densenet169": models.densenet169,
                         "densenet201": models.densenet201}
    
    all_weights = {"densenet121": models.densenet.DenseNet121_Weights.DEFAULT,
                   "densenet169": models.densenet.DenseNet169_Weights.DEFAULT,
                   "densenet201": models.densenet.DenseNet201_Weights.DEFAULT}
    
    assert architecture in all_architectures, f"Should be one of {all_architectures.keys()}"
    
    model = all_architectures[architecture](weights=all_weights[architecture], 
                                           progress = True)
    
    
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)
    
    return model


if __name__== "__main__":
    
    img = torch.randn((5,3, 32,32)) #(NCHW)
    # model = classification_model_resnet("resnet18", 3)
    # model = classification_model_densenet("densenet121", 3)
    model = classification_model_vgg("vgg19_bn", 3).eval()