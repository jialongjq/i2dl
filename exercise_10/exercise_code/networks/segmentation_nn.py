"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ELU()
        self.maxpool = nn.MaxPool2d(3, 3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=8, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.convt3 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        
        self.up1=nn.Upsample(scale_factor=3, mode='nearest')

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.up1(x)
        
        #print(x.size())
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.up1(x)
        
        #print(x.size())
        x = self.convt3(x)
        x = self.relu(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        
        #return torch.squeeze(x, dim=1)
        return x

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
