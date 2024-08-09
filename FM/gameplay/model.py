
import torch
import torch.nn as nn
import argparse
from lib.policy import InverseActionNet, CNN_Transformer_3dconv
import yaml
from collections import deque
import time
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, config_file, batch_size):
        super().__init__()
        with open(config_file) as file:
            configs = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.net = CNN_Transformer_3dconv(**configs)
        self.pred_seq_len = configs['pred_seq_len']
        self.head = nn.Linear(configs["hidsize"], configs["num_class"] * 2)
        self.max_queue_size = configs['input_seq_len']
        self.queue_features = deque()

    def forward(self, x):
        pi_h = self.net(x)[:, -self.pred_seq_len:, :]
        out = self.head(pi_h)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] // 2, 2)
        return out

    @torch.no_grad()
    def extract_cnn_feature(self, x):
        x = self.net.img_preprocess(x)
        x = self.net.img_process(x)
        self.queue_features.append(x.squeeze())
        if len(self.queue_features) > self.max_queue_size:
            self.queue_features.popleft()
        # return x

    @torch.no_grad()
    def transformer_forward(self):
        x = torch.stack(list(self.queue_features), dim=0).unsqueeze(0)
        if self.net.add_pos_embed:
            x = x + self.net.pos_embedding
        x = self.net.transformer_encoder(x)
        x = F.relu(x)
        x = self.net.final_ln(x)
        pi_h = x[:, -self.pred_seq_len:, :]
        out = self.head(pi_h)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] // 2, 2)
        return out

    def reset_queue(self):
        self.queue_features.clear()


if __name__ == '__main__':
    model = Model('model_config.yaml', 1).to('cuda')
    x = torch.randn(2, 32, 128, 128, 3).to('cuda')
    a = model(x)
    print(a.shape)

    for i in range(1000):
        t = time.time()
        img = torch.randn(1, 1, 128, 128, 3).to('cuda')
        model.extract_cnn_feature(img)
        print(len(model.queue_features))
        if len(model.queue_features) == 32:
            output = model.transformer_forward()
            print(output.shape)
        print(time.time() - t)
