import os

import torch.cuda
from torch.utils.data import DataLoader
from torchinfo import summary

from train.data_load import Image_and_force
from train.model import *
from train.parameters import *
from train.apply_img_augmentation import *
from train.predict import *

train_path = './train'
full_dataset = Image_and_force("./data/full_dataset")

def train(model, optimizer, scheduler):
    encoder, decoder, regression_model = model[0], model[1], model[2]

    encoder.train()
    decoder.train()
    regression_model.train()

    MSElosses = []
    L1losses = []

    for batch_idx, (X, y, force) in enumerate(train_loader):
        X = apply_all_frames_RGB_image(X) if USE_RGB_IMG else apply_all_frames(X)
        X, y, force = X.float().to(device), y.to(device), force.to(device)

        optimizer.zero_grad()

        pred_force = prediction(X, model)

        # CEloss = torch.nn.functional.cross_entropy(pred_class.float(), y.float())
        # CElosses.append(CEloss.item())

        MSEloss = torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float())
        L1_loss = torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float())

        MSElosses.append(MSEloss.item())
        L1losses.append(L1_loss.item())

        loss = (MSEloss + L1_loss) / 2

        # CEloss.backward(retain_graph=True)
        MSEloss.backward()
        optimizer.step()

    scheduler.step()
    print("Train: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(np.mean(MSElosses), np.mean(L1losses)))
    return MSElosses


def validation(model, optimizer, epoch):
    encoder, decoder, regression_model = model[0], model[1], model[2]

    encoder.eval()
    decoder.eval()
    regression_model.eval()

    MSEloss = 0
    L1_loss = 0

    with torch.no_grad():
        for X, y, force in val_loader:
            X, y, force = X.float().to(device), y.to(device), force.to(device)

            pred_force = prediction(X, model)

            # CEloss += torch.nn.functional.cross_entropy(pred_class.float(), y.float()).item()
            MSEloss += torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()
            L1_loss += torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float(), reduction='mean').item()

    # CEloss /= len(test_loader)
    MSEloss /= len(test_loader)
    L1_loss /= len(test_loader)

    torch.save(encoder.state_dict(), os.path.join(train_path + '/models/cnn.pth'))
    torch.save(decoder.state_dict(), os.path.join(train_path + '/models/rnn.pth'))
    torch.save(regression_model.state_dict(), os.path.join(train_path + '/models/fc.pth'))

    torch.save(optimizer.state_dict(), os.path.join(train_path + '/models/optim.pth'))

    # print("({}) \t\t{:.5f}\t\t{:.5f}\t\t{:.5f}%".format(epoch, CEloss, MSEloss, test_score * 100))
    print("({}) \t\t{:.5f}\t\t{:.5f}".format(epoch, MSEloss, L1_loss))
    print()

    return MSEloss


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Image_and_force('./'
                                    'data/two/train')
    test_dataset = Image_and_force('./data/two/test')
    val_dataset = Image_and_force('./data/two/val')

    # DataLoader로 변환
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델
    [cnn, rnn, fc], [cnn_params, rnn_params, fc_params], optim, scheduler = initialize_model(device)

    train_losses = []
    test_losses = []

    # Train

    for epoch in range(epochs):
        # Train and Validation model
        train_MSE_losses = train([cnn, rnn, fc],
                                                  optim,scheduler)
        val_MSE_losses = validation([cnn, rnn, fc],
                                                optim, epoch)

        train_losses.append(train_MSE_losses)
        test_losses.append(val_MSE_losses)

        np.save(train_path + './models/train_losses.npy', np.array(train_losses))
        np.save(train_path + './models/test_losses.npy', np.array(test_losses))