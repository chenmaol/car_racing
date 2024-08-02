
import torch
import torch.nn as nn
import argparse
from lib.policy import InverseActionNet, CNN_Transformer_3dconv
import yaml
# torch.autograd.set_detect_anomaly(True)
#
# class Model(nn.Module):
#     def __init__(self, config_file, batch_size):
#         super().__init__()
#         with open(config_file) as file:
#             configs = yaml.load(file.read(), Loader=yaml.FullLoader)
#
#         self.net = InverseActionNet(**configs)
#         # self.net = CNN_Transformer(**configs)
#         self.head = nn.Linear(configs["hidsize"], configs["num_class"] * 2)
#         self.hidden_state = self.net.initial_state(batch_size)
#         self.dummy_first = torch.zeros((configs["timesteps"], 1))
#
#     def forward(self, x):
#         model_input = {"img": x}
#         (pi_h, _), state_out = self.net(model_input, state_in=self.hidden_state, context={"first": self.dummy_first})
#         out = self.head(pi_h)
#         out = out.reshape(out.shape[0], out.shape[1], out.shape[2] // 2, 2)
#         return out


class Model(nn.Module):
    def __init__(self, config_file, batch_size):
        super().__init__()
        with open(config_file) as file:
            configs = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.net = CNN_Transformer_3dconv(**configs)
        self.head = nn.Linear(configs["hidsize"], configs["num_class"] * 2)

    def forward(self, x):
        pi_h = self.net(x)
        out = self.head(pi_h)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] // 2, 2)
        return out


if __name__ == '__main__':
    model = Model('model_config.yaml', 1).to('cuda')
    x = torch.randn(2, 64, 128, 128, 3).to('cuda')
    a = model(x)
    print(a.shape)
