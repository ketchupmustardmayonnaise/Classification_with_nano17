from train.data_load import Image_and_force
from train.model_densenet import *
from train.apply_img_augmentation import *
import matplotlib.pyplot as plt
from train.predict import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 모델
    cnn, rnn, fc = initialize_model(device, train=False)[0]
    cnn.load_state_dict(torch.load('./train/models/encoder.pth'))
    rnn.load_state_dict(torch.load('./train/models/decoder.pth'))
    fc.load_state_dict(torch.load('./train/models/regression_model.pth'))

    cnn.eval()
    rnn.eval()
    fc.eval()

    test_dataset = Image_and_force('./data/two/test')
    test_loader = DataLoader(test_dataset, batch_size=1)

    MSELoss = 0
    L1Loss = 0
    i = 0

    f = []
    pf = []

    for X, y, force in test_loader:
        X = np.expand_dims(X, axis=1)
        X = torch.from_numpy(X).float()
        _force = force.detach().cpu().numpy()

        X, y, force = X.to(device), y.to(device), force.to(device)

        pred_force = prediction(X, [cnn, rnn, fc], train=False)
        _pred_force = pred_force.detach().cpu().numpy()

        _MSEloss = torch.nn.functional.mse_loss(pred_force.float(), force.float())
        _L1loss = torch.nn.functional.l1_loss(pred_force.float(), force.float())

        f.append(_force[0])
        pf.append(_pred_force[0])

        MSELoss += _MSEloss
        L1Loss += _L1loss

    result = []
    shape = [49, 51, 31, 24, 33, 31, 43, 24, 23, 19, 17, 25]
    idx = 0
    i = 0
    for count in shape:
        result.append(f[idx:idx + count])
        idx += count
    f = result

    result = []
    idx = 0
    for count in shape:
        result.append(pf[idx:idx + count])
        idx += count
    pf = result

    for index in range(12):
        _f = f[index]
        _pf = pf[index]
        plt.subplot(3, 4, index + 1)
        plt.plot(np.array(_f), color='r', label='Force')
        plt.plot(np.array(_pf), color='b', label='Predicted Force')

    MSELoss /= len(test_loader)
    L1Loss /= len(test_loader)
    print("Train: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(MSELoss, L1Loss))
    plt.legend()
    plt.show()