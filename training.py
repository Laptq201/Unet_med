import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from custom_data_loader import imageLoader
from models import *
from losses import *

output_file = open('./saved_models/Logs/output.txt', 'w')
error_file = open('./saved_models/Logs/error.txt', 'w')


class DualStream:
    '''
    Mục đích: Ndung được hiển thị trên console 
    và được ghi vào file cùng một lúc không có độ trễ
    - flush(): đảm bảo rằng dữ liệu được ghi 
            vào file ngay lập tức
    '''

    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream

    def write(self, message):
        self.terminal_stream.write(message)
        self.file_stream.write(message)

    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()


# Chuyển hướng stdout vào output file và stderr vào error file
sys.stdout = DualStream(sys.stdout, output_file)
sys.stderr = DualStream(sys.stderr, error_file)


# Set images and masks path

TRAIN_IMG_PATH = "./data/training/images/"
TRAIN_MASK_PATH = "./data/training/masks/"

VAL_IMG_PATH = "./data/validation/images/"
VAL_MASK_PATH = "./data/validation/masks/"
###


train_img_list = sorted(os.listdir(TRAIN_IMG_PATH))
train_mask_list = sorted(os.listdir(TRAIN_MASK_PATH))

train_img_list = [file for file in train_img_list if file.endswith('.npy')]
train_mask_list = [file for file in train_mask_list if file.endswith('.npy')]

val_img_list = sorted(os.listdir(VAL_IMG_PATH))
val_mask_list = sorted(os.listdir(VAL_MASK_PATH))

val_img_list = [file for file in val_img_list if file.endswith('.npy')]
val_mask_list = [file for file in val_mask_list if file.endswith('.npy')]
# Log
print(f"Number of training images: {len(train_img_list)}")
print(f"Number of training masks: {len(train_mask_list)}")


# Xu ly cho BraTS2023
# Find distribution/weights of each class
columns = ['0', '1', '2', '3']
df = []

for img in range(len(train_mask_list)):
    tmp_mask = np.load(TRAIN_MASK_PATH + "/" + train_mask_list[img])
    tmp_mask = np.argmax(tmp_mask, axis=0)
    val, count = np.unique(tmp_mask, return_counts=True)
    conts_dict = dict(zip(val, count))

    df.append(conts_dict)


df = pd.DataFrame(df, columns=columns)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

wt0 = round(total_labels / (n_classes * label_0), 2)
wt1 = round(total_labels / (n_classes * label_1), 2)
wt2 = round(total_labels / (n_classes * label_2), 2)
wt3 = round(total_labels / (n_classes * label_3), 2)

print("=" * 50)
print(f"Weight for class 0: {wt0}")
print(f"Weight for class 1: {wt1}")
print(f"Weight for class 2: {wt2}")
print(f"Weight for class 3: {wt3}")
print("=" * 50)
###

class_weights = {0: wt0, 1: wt1, 2: wt2, 3: wt3}


# Hyperparameters

learning_rate = 1e-4
BATCH_SIZE = 4
accumulation_steps = 30  # Gradient accumulation steps
num_iteration = 10 ^ 4

# setting for run
gpu = 0  # GPU number
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

torch.cuda.empty_cache()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Model
model = AR2B_UNet(in_channels=4, num_classes=4).to(device)

print(f"Model created with {count_params(model)} parameters")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class_weights_list = [class_weights[i] for i in range(len(class_weights))]
loss_function = Loss(num_classes=4, class_weights=class_weights_list,
                     device_num=gpu).to(device)


# Vì tính chất của bộ dataset BraTS nên ta cần phải custom lại hàm
# dataLoader thay vì sử dụng DataLoader mặc định của PyTorch
train_data = imageLoader(TRAIN_IMG_PATH, train_img_list,
                         TRAIN_MASK_PATH, train_mask_list, BATCH_SIZE,
                         num_worker=8)
val_data = imageLoader(VAL_IMG_PATH, val_img_list,
                       VAL_MASK_PATH, val_mask_list, BATCH_SIZE,
                       num_worker=8)


scaler = torch.amp.GradScaler()

print("Start training")

model_path = "./saved_models/AR2B_model.pth"

if os.path.isfile(model_path):
    saved_model = torch.load(model_path)
    model.load_state_dict(saved_model)
    print("Model loaded")


"""
Sen = sensitivity
Spe = specificity
Prec = precision
"""
result_df = pd.DataFrame(columns=['epoch', 'Loss', 'Accuracy',
                                  'Dice Coef(0)', 'Dice Coef(1)', 'Dice Coef(2)', 'Dice Coef(3)',
                                  'Dice Coef Necrotic', 'Dice Coef Edema', 'Dice Coef Enhancing',
                                  'Sen(0)', 'Sen(1)', 'Sen(2)', 'Sen(3)',
                                  'Spe(0)', 'Spe(1)', 'Spe(2)', 'Spe(3)',
                                  'Prec(0)', 'Prec(1)', 'Prec(2)', 'Prec(3)'
                                  ])

best_loss = float('inf')
num_classes = 4

for epoch in range(num_iteration):
    model.train()
    epoch_loss = 0

    dice_score_necrotic = 0
    dice_score_edema = 0
    dice_score_enhancing = 0

    # metrics initialization
    total_accuracy = 0
    total_sensitivity = [0] * num_classes
    total_specificity = [0] * num_classes
    total_precision = [0] * num_classes
    total_dice_score = [0] * num_classes

    for batch_num, (imgs, masks) in enumerate(train_data):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast():

            prob = model(imgs)  # Ouput của model sẽ là probablity vì
            # lúc sau ta sẽ dùng cross entropy loss (có trong Loss)
            outputs = torch.argmax(prob, dim=1)
            masks = torch.argmax(prob, dim=1)

            loss = loss_function(prob, masks)
            loss = loss.mean()

        # Raise error if loss is NaN
        if np.isnan(loss.item()):
            img_names = train_data.dataset.img_list[batch_num *
                                                    BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
            mask_names = train_data.dataset.mask_list[batch_num *
                                                      BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]

            raise ValueError(f"NaN loss found in batch {batch_num}\n"
                             f"Image names: {img_names}\n"
                             f"Mask names: {mask_names}\n")

        scaler.scale(loss).backward(retain_graph=True)

        if (batch_num + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        print(f"Epoch: {epoch}, Batch: {batch_num}, Loss: {loss.item()}")

        epoch_loss += loss.item()

        total_accuracy += accuracy(outputs, masks).item()
        for i in range(num_classes):
            total_sensitivity[i] += sensitivity(outputs, masks, i).item()
            total_specificity[i] += specificity(outputs, masks, i).item()
            total_precision[i] += precision(outputs, masks, i).item()
            total_dice_score[i] += dice_score(outputs, masks, i).item()

        dice_score_necrotic += dice_coef_necrotic(outputs, masks).item()
        dice_score_edema += dice_coef_edema(outputs, masks).item()
        dice_score_enhancing += dice_coef_enhancing(outputs, masks).item()

    # get the average of the metrics
    epoch_loss /= len(train_data)
    total_accuracy /= len(train_data)
    total_sensitivity = [sensitivity /
                         len(train_data) for sensitivity in total_sensitivity]
    total_specificity = [specificity /
                         len(train_data) for specificity in total_specificity]
    total_precision = [precision / len(train_data)
                       for precision in total_precision]
    total_dice_score = [dice_score / len(train_data)
                        for dice_score in total_dice_score]
    dice_score_necrotic /= len(train_data)
    dice_score_edema /= len(train_data)
    dice_score_enhancing /= len(train_data)

    # logging
    print("=" * 50)
    print((
        f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {total_accuracy}, Dice Coef: {total_dice_score}, "
        f"Dice Coef Necrotic: {dice_score_necrotic}, Dice Coef Edema: {dice_score_edema}, "
        f"Dice Coef Enhancing: {dice_score_enhancing}, Sensitivity: {total_sensitivity}, "
        f"Specificity: {total_specificity}, "
        f"Precision: {total_precision}"
    ))
    print("=" * 50)

    # New row to DataFrame
    new_row = {'Epoch': epoch + 1,
               'Loss': epoch_loss,
               'Accuracy': total_accuracy,
               'Dice Coef (0)': total_dice_score[0],
               'Dice Coef (1)': total_dice_score[1],
               'Dice Coef (2)': total_dice_score[2],
               'Dice Coef (3)': total_dice_score[3],
               'Dice Coef Necrotic': dice_score_necrotic,
               'Dice Coef Edema': dice_score_edema,
               'Dice Coef Enhancing': dice_score_enhancing,
               'Sensitivity (0)': total_sensitivity[0],
               'Sensitivity (1)': total_sensitivity[1],
               'Sensitivity (2)': total_sensitivity[2],
               'Sensitivity (3)': total_sensitivity[3],
               'Specificity (0)': total_specificity[0],
               'Specificity (1)': total_specificity[1],
               'Specificity (2)': total_specificity[2],
               'Specificity (3)': total_specificity[3],
               'Precision (0)': total_precision[0],
               'Precision (1)': total_precision[1],
               'Precision (2)': total_precision[2],
               'Precision (3)': total_precision[3]}

    results_df = pd.concat([results_df, pd.DataFrame(
        new_row, index=[0])], ignore_index=True)

    #####
    # Validation
    #####

    if (epoch + 1) % 50 == 0:
        model.eval()  # Set the model to evaluation mode

        # Initialize metrics for validation
        val_loss = 0
        val_total_accuracy = 0
        val_total_sensitivity = [0] * num_classes
        val_total_specificity = [0] * num_classes
        val_total_precision = [0] * num_classes
        val_total_dice_score = [0] * num_classes

        val_dice_score_necrotic = 0
        val_dice_score_edema = 0
        val_dice_score_enhancing = 0
        # ... other metrics initialization ...

        with torch.no_grad():  # Disable gradients for validation
            for val_batch_num, (val_imgs, val_masks) in enumerate(val_data):
                val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

                val_logits = model(val_imgs)
                val_outputs = torch.argmax(val_logits, dim=1)
                val_masks = torch.argmax(val_masks, dim=1)

                # Compute the validation loss
                val_loss += loss_function(val_logits, val_masks).mean().item()

                # Calculate metrics for validation
                val_total_accuracy += accuracy(val_outputs, val_masks).item()
                for class_id in range(num_classes):
                    val_total_sensitivity[class_id] += sensitivity(
                        val_outputs, val_masks, class_id).item()
                    val_total_specificity[class_id] += specificity(
                        val_outputs, val_masks, class_id).item()
                    val_total_precision[class_id] += precision(
                        val_outputs, val_masks, class_id).item()
                    val_total_dice_score[class_id] += dice_score(
                        val_outputs, val_masks, class_id).item()

                val_dice_score_necrotic += dice_coef_necrotic(
                    val_masks, val_outputs).item()
                val_dice_score_edema += dice_coef_edema(
                    val_masks, val_outputs).item()
                val_dice_score_enhancing += dice_coef_enhancing(
                    val_masks, val_outputs).item()

                # ... other per-class metrics ...

        # Calculate average validation metrics
        val_loss /= len(val_data_gen)
        val_total_accuracy /= len(val_data)
        val_total_sensitivity = [
            sens / len(val_data) for sens in val_total_sensitivity]
        val_total_specificity = [
            spec / len(val_data) for spec in val_total_specificity]
        val_total_precision = [sens / len(val_data)
                               for sens in val_total_precision]
        val_total_dice_score = [
            spec / len(val_data) for spec in val_total_dice_score]
        val_dice_score_necrotic /= len(train_data)
        val_dice_score_edema /= len(train_data)
        val_dice_score_enhancing /= len(train_data)
        # ... other metrics ...

        # Print validation results
        print(f"-----------Validation Epoch {epoch+1}, Loss: {val_loss}, Accuracy: {val_total_accuracy}, Dice Coef: {val_total_dice_score}, "
              f"Dice Coef Necrotic: {val_dice_score_necrotic}, Dice Coef Edema: {val_dice_score_edema}, "
              f"Dice Coef Enhancing: {val_dice_score_enhancing}, Sensitivity: {val_total_sensitivity}, "
              f"Specificity: {val_total_specificity}, "
              f"Precision: {val_total_precision}")
        # ... other metrics print statements ...

        # Append results to DataFrame
        new_row = {'Epoch': "Validation " + str(epoch + 1),
                   'Loss': val_loss,
                   'Accuracy': val_total_accuracy,
                   'Dice Coef (0)': val_total_dice_score[0],
                   'Dice Coef (1)': val_total_dice_score[1],
                   'Dice Coef (2)': val_total_dice_score[2],
                   'Dice Coef (3)': val_total_dice_score[3],
                   'Dice Coef Necrotic': val_dice_score_necrotic,
                   'Dice Coef Edema': val_dice_score_edema,
                   'Dice Coef Enhancing': val_dice_score_enhancing,
                   'Sensitivity (0)': val_total_sensitivity[0],
                   'Sensitivity (1)': val_total_sensitivity[1],
                   'Sensitivity (2)': val_total_sensitivity[2],
                   'Sensitivity (3)': val_total_sensitivity[3],
                   'Specificity (0)': val_total_specificity[0],
                   'Specificity (1)': val_total_specificity[1],
                   'Specificity (2)': val_total_specificity[2],
                   'Specificity (3)': val_total_specificity[3],
                   'Precision (0)': val_total_precision[0],
                   'Precision (1)': val_total_precision[1],
                   'Precision (2)': val_total_precision[2],
                   'Precision (3)': val_total_precision[3]}

        results_df = pd.concat([results_df, pd.DataFrame(
            new_row, index=[0])], ignore_index=True)
