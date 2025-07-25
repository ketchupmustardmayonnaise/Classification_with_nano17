import numpy as np
from torch.utils.data import Dataset
import os
import zipfile

class RFTDataset(Dataset):

    def __init__(self, path='./../dataset/rft64'):
        self.img = []
        self.force = []
        self.label = []

        self.img_ts = []
        self.force_ts = []
        self.frame = []

        for label in os.listdir(path): # 각 label에 대하여
            for file_name in os.listdir(path + '/' + label): # 각 파일들에 대하여
                with zipfile.ZipFile(path + '/' + label + '/' + file_name, 'r') as zf: # 각 zip파일들에 대하여
                    for npzfile in zf.namelist(): # 각 npz파일들에 대하여
                        with zf.open(npzfile) as nf:
                            data = np.load(nf)
                            self.img.append(data['img'])
                            self.force.append(data['force'])
                            self.img_ts.append(data['img_ts'])
                            self.force_ts.append(data['force_ts'])
                            self.label.append(int(label))
                            self.frame.append(len(data['img']))

        print(sum(self.frame))

RFTDataset()