from data_load import Image_and_force
from model_densenet import *
from apply_img_augmentation import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 모델
    model = DenseNet121()
    model.load_state_dict(torch.load('./model/encoder.pth'))
    model = model.to(device)

    test_dataset = Image_and_force('./../data/two/test')

    MSELoss = 0
    L1Loss = 0

    for i in range(test_dataset.__len__()):
        X, y, _force = test_dataset.__getitem__()
        X = np.expand_dims(X, axis=1)
        X = torch.from_numpy(X).float()
        force = torch.from_numpy(_force).float()

        X, y, force = X.to(device), y.to(device), force.to(device)

        pred_force = model(X)
        _pred_force = pred_force.detach().cpu().numpy()

        _MSEloss = torch.nn.functional.mse_loss(pred_force.reshape((len(force))).float(), force.float())
        _L1loss = torch.nn.functional.l1_loss(pred_force.reshape((len(force))).float(), force.float())

        plt.subplot(3, 4, i + 1)
        plt.plot(_force, color='r', label='Force')
        plt.plot(_pred_force, color='b', label='Predicted Force')

        print("Train: \tMSE loss: {:.5f}\tL1 loss: {:.5f}".format(_MSEloss, _L1loss))
        MSELoss += _MSEloss
        L1Loss += _L1loss

    MSELoss /= 12
    L1Loss /= 12
    print(MSELoss, L1Loss)
    plt.legend()
    plt.show()