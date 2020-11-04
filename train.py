"""
This script will do 3 things:
    1. Augment train images
    2. Train model and save model based on validation
    3. Inference: will output a predictions in predictions folder

Sample usage:
    python train.py -e 100 -s 0
"""
import os
import time

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse

from src.dataset import cs5242_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=100, type=int, help="number of epochs to train")
parser.add_argument("-s", "--seed", default=0, type=int, help="random seed for pytoch and numpy")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)



"""
1. Augment data with LEFT RIGHT FLIP
"""
TRAIN_IMG_DIR = 'data/nus-cs5242/train_image/train_image'
TRAIN_LABELS_DIR = 'data/nus-cs5242/train_label.csv'

img_name_list = os.listdir(TRAIN_IMG_DIR)
img_labels = pd.read_csv(TRAIN_LABELS_DIR, index_col=0)
im_to_flip = img_labels.loc[img_labels.Label !=1, :].index
print(f"number of img to train on: {len(img_labels)}")

# IF-block checks that data is not augmented
if len(img_labels) <= 1164:
    print("Augmenting data")
    len_flipped_img = 1
    for img_index in im_to_flip:
        
        # open image
        img_name = str(img_index) + '.png'
        orig_img = Image.open(TRAIN_IMG_DIR + '/' + img_name)
        
        # flip image
        flipped_img = orig_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # save flipped image in train folder
        flipped_idx = len(img_labels) + len_flipped_img
        flipped_img.save( TRAIN_IMG_DIR + '/' + str(flipped_idx) + '.png' )
        
        # save label in labels df
        img_labels.loc[flipped_idx,'Label'] = int(img_labels.loc[img_index,'Label'])

        len_flipped_img+=1

    # save flipped labels
    img_labels.Label = img_labels.Label.apply(int)
    img_labels.to_csv(TRAIN_LABELS_DIR)
    print(f"number of img to train on after augmentation: {len(img_labels)}")



"""
2. TRAIN MODEL
"""
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.RandomRotation(degrees=15),
        transforms.RandomVerticalFlip(),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset = 'data/nus-cs5242/'
label_directory = os.path.join(dataset, 'train_label.csv')
train_directory = os.path.join(dataset, 'train_image/train_image')
# valid_directory = os.path.join(dataset, 'test_image/train_image')

batch_size, num_classes, split = 32, 3, 0.8
data = {
    'train': cs5242_dataset(img_dir=train_directory, txt_path = label_directory, transform = image_transforms['train'], train=True, split=split),
    'valid': cs5242_dataset(img_dir=train_directory, txt_path = label_directory, transform = image_transforms['train'], train=False, split=split),
    # 'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    # 'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

train_data_size, valid_data_size= len(data['train']), len(data['valid'])
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
print(f"train_data_size, valid_data_size: {train_data_size, valid_data_size}")


# model
resnet50 = models.resnet50(pretrained=True)

# for param in resnet50.parameters():
#     param.requires_grad = False

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)

loss_func = nn.NLLLoss()
optimizer = optim.RMSprop(lr=0.0001, params=resnet50.parameters())


def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = []
    best_acc, best_epoch = 0.0, 0
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs('models/'+timestamp)

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()
        train_loss, train_acc, valid_loss, valid_acc = 0.0, 0.0, 0.0, 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save(model, 'models/' + timestamp + '/' + '_model_'+str(epoch+1)+'.pt')

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    torch.save(history, 'models/'+timestamp +'/'+'_history.pt')
    return model, history

trained_model, history = train_and_valid(resnet50, loss_func, optimizer, args.epochs)

# history = np.array(history)
# plt.plot(history[:, 0:2])
# plt.legend(['Tr Loss', 'Val Loss'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.ylim(0, 1)
# plt.savefig(dataset+'_loss_curve.png')
# plt.show()

# plt.plot(history[:, 2:4])
# plt.legend(['Tr Accuracy', 'Val Accuracy'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)
# plt.savefig(dataset+'_accuracy_curve.png')
# plt.show()



"""
3. Make Prediction
"""
device = torch.device('cpu')
transform=transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])

# dataset = 'data/nus-cs5242/'
# test_directory = os.path.join(dataset, 'test_image/test_image')
test_directory = 'data/nus-cs5242/test_image'
data = {
    'test': datasets.ImageFolder(root=test_directory, transform=transform),
}

# batch_size = 32
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], shuffle=False)  # , batch_size=batch_size
print(f"test_data_size: {test_data_size}")

trained_model = trained_model.to(device)
torch.no_grad()
trained_model.eval()
csv_data = [("ID","Label")]

print("running predictions")
for j, (inputs, labels) in enumerate(test_data):
    inputs = inputs.to(device)
    output = trained_model(inputs)
    _, predicted = torch.max(output, 1)
    csv_data.append((str(j),str(int(predicted))))

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
predictions_filename = f'predictions/{timestamp}.csv'
f = open(predictions_filename,'w',newline='')
writer = csv.writer(f)
for i in csv_data:
    writer.writerow(i)
f.close()

print(f"DONE: Predictions saved in {predictions_filename}")