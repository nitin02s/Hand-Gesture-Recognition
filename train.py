from data_loader import CustomDataset
import argparse
import logging
import sys
from pathlib import Path
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import Classifier
from data_loader import CustomDataset
from torchvision import transforms
from torcheval.metrics.classification import MulticlassRecall
from torcheval.metrics import MulticlassPrecision
import torchmetrics
# from torchinfo import summary
import numpy as np


classes = ("one","two","three","palm","fist","call","ok","like")
csv_dir = Path('data_1/final_data.csv') 
img_dir = Path('data_1/imgs')
mean = [0.5401, 0.5045, 0.4845]
std = [0.2259, 0.2270, 0.2279]
#directory to save checkpoint
dir_checkpoint = Path('weights/resnet_101')

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 32,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    

    data_transforms = transforms.Compose([
    transforms.Resize((int(1920*scale),int(1440*scale))),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5401, 0.5045, 0.4845], [0.2259, 0.2270, 0.2279])
    ])

    # 1. Create dataset
    dataset = CustomDataset(csv_dir, root_dir=img_dir, transform=data_transforms)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='Hand')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')


    # metric = StreamSegMetrics(net.n_classes)
    precision = MulticlassPrecision(num_classes=5)
    recall = MulticlassRecall(num_classes=5)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-7, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #adam optimizer
    # optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Cross entropy loss
    criterion = nn.CrossEntropyLoss()
    log_dict = {
        'training_loss_per_batch': [],
        'validation_loss_per_batch': [],
        'training_accuracy_per_epoch': [],
        'validation_accuracy_per_epoch': [],
        'validation_precision_per_epoch': [],
        'validation_recall_per_epoch': [],
        'validation_F1_per_epoch': []

    } 
    num_classes = 5
    val_loss = []
    train_loss = []
    train_acc = []
    #focal loss
    global_step = 0
    transform = T.ToPILImage()
    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        train_loss = []
        total_correct = 0
        total_instances = 0
        correct_predictions = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

    
                images = batch['image']
                points = batch['points']
                true_class = batch['class']

                images = images.to(device=device, dtype=torch.float32)
                points = points.to(device=device, dtype=torch.float32)
                # true_class = torch.eye(num_classes,dtype=torch.float32)[true_class].squeeze()
                true_class = true_class.to(device=device)
                # print(true_class)
                with torch.cuda.amp.autocast():
                    pred = net(points,images)
                    # print(true_class.shape,pred.shape)
                    loss = criterion(pred,true_class)
        
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_loss.append(loss.item())
                experiment.log({
                    'train loss' : loss.item()
                })
                predictions = torch.argmax(pred, dim=1)
                correct_predictions = sum(predictions==true_class).item()
                total_correct+=correct_predictions
                # predictions = predictions.to('cpu')
                # true_class = true_class.to('cpu')
                # precision.update(predictions, true_class)
                # recall.update(predictions, true_class)
                total_instances+=len(images)
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                print({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # train_acc.append(100 * correct / total)
        final_precision = precision.compute()
        final_recall = recall.compute()
        train_accuracy = round(total_correct/total_instances, 3)    
        log_dict['training_accuracy_per_epoch'].append(train_accuracy) 
        print("training accuracy :", train_accuracy)

        print('validating...')
        val_losses = []
           
        net.eval()
        total_instances = 0
        correct_predictions = 0
        predicted_true = 0
        correct_true = 0
        total_correct = 0
        y_pred = []
        y_true = []
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

                images = batch['image']
                points = batch['points']
                true_class = batch['class']
                images = images.to(device=device, dtype=torch.float32)
                points = points.to(device=device, dtype=torch.float32)
                true_class = true_class.to(device=device)

                #  making predictions
                predictions = net(points,images)
                #  computing loss
                val_loss = criterion(predictions, true_class)
                #experiment.log({
                #    'validation loss' : val_loss.item()
                #})
                log_dict['validation_loss_per_batch'].append(val_loss.item())
                val_losses.append(val_loss.item())

                #  computing accuracy
                print('deriving validation accuracy...')
                predictions = torch.argmax(predictions, dim=1)
                predictions = predictions.to('cpu')
                y_pred.extend(predictions)
                
                true_class = true_class.to('cpu')
                y_true.extend(true_class)
                correct_predictions = sum(predictions==true_class).item()
                total_correct+=correct_predictions
                precision.update(predictions, true_class)
                recall.update(predictions, true_class)
                # predicted_classes = torch.argmax(predictions, dim=1)
                # predicted_true += torch.sum(predictions).float()
                # correct_true += torch.sum(predictions == correct_predictions * predictions == true_class).float()
                total_instances+=len(images)

        val_accuracy = round(total_correct/total_instances, 3)    
        final_precision = precision.compute()
        final_recall = recall.compute()
        f1_score = 2 * final_precision * final_recall / (final_precision + final_recall)
        scheduler.step(val_accuracy) 
        print("validation accuracy :", val_accuracy) 
        print("recall :", final_recall)     
        print("precision :", final_precision)
        print("F1 Score :", f1_score)
        log_dict['validation_accuracy_per_epoch'].append(val_accuracy)    
        log_dict['validation_precision_per_epoch'].append(final_precision)
        log_dict['validation_recall_per_epoch'].append(final_recall)
        log_dict['validation_F1_per_epoch'].append(f1_score)
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output_mobv3_small.png')

                # print(f'train acc: {(correct / total):.4f}')
        

        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'training accuracy' : log_dict['training_accuracy_per_epoch'][-1],
            'validation accuracy': log_dict['validation_accuracy_per_epoch'][-1],
            'validation precision': log_dict['validation_precision_per_epoch'][-1],
            'validation recall': log_dict['validation_recall_per_epoch'][-1],
            'validation F1': log_dict['validation_F1_per_epoch'][-1],
            


            # 'images': wandb.Image(images[0].cpu()),
            # 'masks': {
            #     'true': wandb.Image(transform(true_masks[0].float().cpu())),
            #     'pred': wandb.Image(transform(masks_pred.argmax(dim=1)[0].float().cpu())),
            # },
            'step': global_step,
            'epoch': epoch,
        })
        precision.reset()
        recall.reset()
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':


    # Hyperparameters to set for training

    bilinear= False
    epochs = 30
    batch_size = 60
    lr = 0.0002
    scale = 0.3
    val = 20.0
    amp = True

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #loading model
    net = Classifier()
    
    #pre loaded model
    # if args.load:
    # net.load_state_dict(torch.load(r"weights\run_2\checkpoint_epoch30.pth", map_location=device))
    # print('Model pre loaded')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=val / 100,
                  amp=amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

