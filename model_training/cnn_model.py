import torch
import torch.nn as nn
from torchvision import models

def get_property_model(num_classes=3, pretrained=True):
    """
    Crea un modelo basado en ResNet50 para clasificar imágenes de propiedades.
    """
    # Descargar ResNet50 pre-entrenada en ImageNet
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except:
        model = models.resnet50(pretrained=pretrained)
        
    # Opcional: Congelar las capas convolucionales base si se quiere un entrenamiento rápido
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Cambiar la capa final (Fully Connected) para adaptarla a nuestras 3 categorías
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model
