import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.ln3 = nn.Linear(42, 64)
        self.ln4 = nn.Linear(64, 64)
        self.ln6 = nn.Linear(64, 64)
        self.ln8 = nn.Linear(1064,8)
        # self.ln8 = nn.Linear(1000,8)

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
        
        self.model = models.densenet201(weights='DEFAULT')
        # self.model = models.resnet101(weights='DEFAULT')

    def forward(self,kp,img):
    	#forward HBB through MobiletNetV2
        img=self.model(img)
        
        #forward keypoints through FC
        kp=self.ln3(kp)
        kp=self.relu(kp)
        kp=self.dropout2(kp)
        kp=self.ln4(kp)
        kp=self.relu(kp)
        kp=self.dropout3(kp)
        kp=self.ln6(kp)
        kp=self.relu(kp)

        # print(kp.shape)
	
        #concatenation
        out= torch.cat([kp, img], dim=1)
        
        out=self.ln8(out)
        out=self.soft(out)
                            
        return out

