

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from model import Classifier
from mp_sc import landmarks
import cv2
import time
import os
import mediapipe as mp
from torcheval.metrics import MulticlassAccuracy
#add directory of the weights
model_path = r"weights\checkpoint_epoch30.pth"
img_dir = r"demo_images"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier()
model.to(device)
model.load_state_dict(torch.load(model_path,map_location=device))

def write_video(orig,output_orig):
    output_orig.write(orig)

def norm_points(points,width,height):

    x = 1/width
    y = 1/height

    pair = [x,y]
    arr = np.tile(pair,21)
    norm = arr*points
    return norm

def predict(image,points):
    points = np.array(points, dtype=np.float32)
    data_transforms = transforms.Compose([
    transforms.Resize((int(1920*0.3),int(1440*0.3))),
    transforms.ToTensor(),
    transforms.Normalize([0.5401, 0.5045, 0.4845], [0.2259, 0.2270, 0.2279])
    ])
    images = data_transforms(image)
    with torch.no_grad():
        model.eval()
        # images = data_transforms(image)
        image = images.unsqueeze(0)
        images = image.to(device=device, dtype=torch.float32)
        points = torch.from_numpy(points)
        points = points.to(device=device, dtype=torch.float32)
        points = points.unsqueeze(0)
        pred = model(points,images)
        pred = pred.cpu().detach().numpy().squeeze()
        output_probs = np.exp(pred)*100
        np.set_printoptions(suppress=True)
        output=pred.argmax()
        k = output.item()
        return k
        

pred_ouput = ["one","two","three","palm","fist","call","ok","like"]

image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(img_dir, image_file)
    image = cv2.imread(image_path)
    img = image
    points, ptsimage = landmarks(image)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    width, height = image.size

    if len(points) > 0:
        points = norm_points(points, width, height)
        pred_class = predict(image,points)
        img = cv2.putText(img, pred_ouput[pred_class], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 3, cv2.LINE_AA)
    else:
        continue

    img = cv2.resize(img,(700,650))
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
