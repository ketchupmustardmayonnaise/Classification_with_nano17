import test
import unittest
from hydra import compose, initialize
from force_estimation_refactoring.models import *
from force_estimation_refactoring.data.capacitive_touchscreen_datamodule import CapacitiveImageDataModule


class MyTestCase(unittest.TestCase):

    def setUp(self):
        # 1. config_path="." 는 현재 파이썬 파일과 같은 위치를 의미합니다.
        # version_base=None을 넣어주어야 에러가 발생하지 않습니다.
        with initialize(version_base=None, config_path="."):
            # 2. 파일 확장자(.yaml)를 제외한 파일 이름만 지정합니다.
             self.cfg = compose(config_name="temporal_feature_extractor")


    def test_dataset(self):
        # 로드된 설정 확인 (디버깅용)
        # print(self.cfg)

        cd = CapacitiveImageDataModule(self.cfg)
        cd.setup(stage='fit')

        loader = cd.train_dataloader()
        train_data = next(iter(loader))

        cd.setup(stage='test')
        loader = cd.test_dataloader()
        test_data = next(iter(loader))

        self.assertEqual(train_data.keys(), test_data.keys())
        self.assertEqual(len(test_data['img']), 1)
        self.assertEqual(train_data['img'].shape[1:], test_data['img'].shape[1:])
        self.assertEqual(train_data['force'].shape[1:], test_data['force'].shape[1:])
        print(train_data.keys())

        """
        img (batch, timestemp, channel, height, width)
        force (batch, timestep, 3)
        label (batch, 8)
        sequence_idx (8,)
        subject_idx (8) (list)
        
        test라면 batch size = 1
        """


    def test_loading_vf_net(self):
        self.vf_net = VisionFeatureExtractor(self.cfg.model.encoder.cnn_name, self.cfg.model.encoder.image_size)
        fake_data = torch.zeros(8, 30, 3, 6, 6)
        output = self.vf_net.forward(fake_data)
        self.assertEqual(output.shape, (8, 30, self.vf_net.size_of_output))


    def test_loading_transformer(self):
        vf_net = VisionFeatureExtractor(self.cfg.model.encoder.cnn_name, self.cfg.model.encoder.image_size)
        tf_net = Transformer(size_of_input=vf_net.size_of_output,
                             num_rnn_layer=self.cfg.model.transformer.num_layers,
                             num_of_heads=self.cfg.model.lstm.num_of_head,

                             dropout=self.cfg.model.dropout,
                             length_of_sequence=self.cfg.data.length_of_window,

                             dim_feedforward=self.cfg.model.transformer.dim_feedforward,
                             export_multistep=self.cfg.model.transformer.export_multistep)

        fake_data = torch.zeros(self.cfg.data.batch_size, 30, 3, 6, 6)
        output = self.vf_net.forward(fake_data)
        output = tf_net.forward(output)
        self.assertEqual(output.shape, (8, 30, 3))


    # def


if __name__ == '__main__':
    unittest.main()