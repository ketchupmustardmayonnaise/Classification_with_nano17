import copy
import os

import cv2
import torch.cuda
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from data_load import Image_and_force, TARGET_INDICES, SIZE, FLAG_RESIZE
from sklearn.metrics import accuracy_score
from torchvision import transforms
from model import *
from scheduler import *
import albumentations as A

epochs = 100000
batch_size = 512
learning_rate = 1e-4
weight_decay = 0.005
log_interval = 8

train_path = './train'
full_dataset = Image_and_force("./data/full_dataset")


def image_transform(X):
    # Augmentation
    transforms = A.ReplayCompose([
        A.Rotate(
            limit=[-180, 180],
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE,
            rotate_method="largest_box",
            crop_border=False
        ),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5)
    ])

    data_size = len(X)
    X_transformed = []

    for index in range(len(X)):
        X_temp = []
        transformed = transforms(image=np.array(X[index][0]))
        replay = transformed['replay']
        X_temp.append(torch.from_numpy(np.array(transformed['image'])).reshape((10, 8)).tolist())

        for frame in range(1, TARGET_INDICES):
            transformed = A.ReplayCompose.replay(replay, image=np.array(X[index][frame]))
            X_temp.append(torch.from_numpy(np.array(transformed['image'])).reshape((10, 8)).tolist())

        X_transformed.append(X_temp)

    if FLAG_RESIZE:
        return torch.Tensor(X_transformed).reshape((data_size, TARGET_INDICES, SIZE[0], SIZE[1])).float()
    else:
        return torch.Tensor(X_transformed).reshape((data_size, TARGET_INDICES, 10, 8)).float()


def prediction(input, model):
    encoder, decoder, classification_model, regression_model = model[0], model[1], model[2], model[3]

    size_x = len(input)
    if FLAG_RESIZE:
        X = input.reshape((size_x * TARGET_INDICES, 1, SIZE[0], SIZE[1]))
    else:
        X = input.reshape((size_x * TARGET_INDICES, 1, 10, 8))
    X = encoder(X)
    X = X.reshape((size_x, TARGET_INDICES, -1))
    X = decoder(X)
    pred_force = regression_model(X)

    return pred_force

def train(log_interval, model, optimizer, scheduler):
    encoder, decoder, classification_model, regression_model = model[0], model[1], model[2], model[3]
    enc_dec_optim, classification_optim, regression_optim = optimizer[0], optimizer[1], optimizer[2]
    enc_dec_scheduler, regression_scheduler = scheduler[0], scheduler[1]

    encoder.train()
    decoder.train()
    classification_model.train()
    regression_model.train()

    CElosses = []
    MSElosses = []

    for batch_idx, (X, y, force) in enumerate(train_loader):
        #X = image_transform(X)
        X, y, force = X.to(device), y.to(device), force.to(device)

        enc_dec_optim.zero_grad()
        classification_optim.zero_grad()
        regression_optim.zero_grad()

        pred_force = prediction(X, model)

        # CEloss = torch.nn.functional.cross_entropy(pred_class.float(), y.float())
        # CElosses.append(CEloss.item())

        criterion = nn.MSELoss()
        MSEloss = torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float())
        MSElosses.append(MSEloss.item())

        # CEloss.backward(retain_graph=True)
        MSEloss.backward()

        enc_dec_optim.step()
        regression_optim.step()

        enc_dec_scheduler.step()
        regression_scheduler.step()
        # classification_optim.step()

    print("Train: \tMSE loss: {:.5f}".format( np.mean(MSElosses)))
    return MSElosses


def validation(model, optimizer, epoch):
    encoder, decoder, classification_model, regression_model = model[0], model[1], model[2], model[3]
    enc_dec_optim, classification_optim, regression_optim = optimizer[0], optimizer[1], optimizer[2]

    encoder.eval()
    decoder.eval()
    classification_model.eval()
    regression_model.eval()

    CEloss = 0
    MSEloss = 0

    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for X, y, force in test_loader:
            X, y, force = X.to(device), y.to(device), force.to(device)

            pred_force = prediction(X, model)

            # CEloss += torch.nn.functional.cross_entropy(pred_class.float(), y.float()).item()
            MSEloss += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()

            # y = torch.max(y, 1)[1]
            # y_pred = pred_class.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            # all_y.extend(y)
            # all_y_pred.extend(y_pred)

    # CEloss /= len(test_loader)
    MSEloss /= len(test_loader)

    # compute accuracy
    # all_y = torch.stack(all_y, dim=0)
    # all_y_pred = torch.stack(all_y_pred, dim=0)
    # test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    torch.save(encoder.state_dict(), os.path.join(train_path + '/models/encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(train_path + '/models/decoder.pth'))
    torch.save(classification_model.state_dict(), os.path.join(train_path + '/models/classification_model.pth'))
    torch.save(regression_model.state_dict(), os.path.join(train_path + '/models/regression_model.pth'))

    torch.save(enc_dec_optim.state_dict(), os.path.join(train_path + '/models/enc_dec_optim.pth'))
    torch.save(classification_optim.state_dict(), os.path.join(train_path + '/models/classification_optim.pth'))
    torch.save(regression_optim.state_dict(), os.path.join(train_path + '/models/regression_optim.pth'))

    # print("({}) \t\t{:.5f}\t\t{:.5f}\t\t{:.5f}%".format(epoch, CEloss, MSEloss, test_score * 100))
    print("({}) \t\t{:.5f}".format(epoch, MSEloss))
    print()

    return MSEloss



if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Image_and_force('./data/two/train')
    test_dataset = Image_and_force('./data/two/test')
    val_dataset = Image_and_force('./data/two/val')

    # DataLoader로 변환
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델
    resnet = resnet_learning()
    resnet = resnet.to(device)
    decoder = DecoderRNN()
    decoder = decoder.to(device)
    classification_model = PredictLabelModel()
    classification_model = classification_model.to(device)
    regression_model = PredictSheerModel()
    regression_model = regression_model.to(device)

    # parameter
    enc_dec_params = list(resnet.parameters()) + list(decoder.parameters())
    classification_params = list(classification_model.parameters())
    regression_params = list(regression_model.parameters())

    # Optimizer
    enc_dec_optim = torch.optim.RMSprop(enc_dec_params, learning_rate, weight_decay=weight_decay)
    classification_optim = torch.optim.RMSprop(classification_params, learning_rate, weight_decay=weight_decay)
    regression_optim = torch.optim.RMSprop(regression_params, learning_rate, weight_decay=weight_decay)

    # Scheduler
    enc_scheduler = CosineAnnealingWarmUpRestarts(enc_dec_optim, T_0=150, eta_max=0.1, T_up=10, gamma=0.5)
    regression_scheduler = CosineAnnealingWarmUpRestarts(enc_dec_optim, T_0=150, eta_max=0.1, T_up=10, gamma=0.5)

    train_losses = []
    test_losses = []

    # Train

    for epoch in range(epochs):
        # Train and Validation model
        train_MSE_losses = train(log_interval, [resnet, decoder, classification_model, regression_model],
                                                  [enc_dec_optim, classification_optim, regression_optim],
                                                    [enc_scheduler, regression_scheduler])
        val_MSE_losses = validation([resnet, decoder, classification_model, regression_model],
                                                [enc_dec_optim, classification_optim, regression_optim],
                                                   epoch)

        train_losses.append(train_MSE_losses)
        test_losses.append(val_MSE_losses)

        np.save(train_path + './train_losses.npy', np.array(train_losses))
        np.save(train_path + './test_losses.npy', np.array(test_losses))