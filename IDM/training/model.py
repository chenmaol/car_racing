import torch
import torch.nn as nn
import argparse
from lib.policy import InverseActionNet


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.net = InverseActionNet(**vars(args)).to(self.device)
        self.head = nn.Linear(args.hidsize, args.num_class * 2).to(self.device)
        self.hidden_state = self.net.initial_state(1)
        self.dummy_first = torch.zeros((args.timesteps, 1)).to(self.device)

    def forward(self, x):
        model_input = {"img": x}
        (pi_h, _), state_out = self.net(model_input, state_in=self.hidden_state, context={"first": self.dummy_first})
        out = self.head(pi_h)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2] // 2, 2)
        return out


if __name__ == '__main__':
    net_args = {'attention_heads': 32,
                'attention_mask_style': 'none',
                'attention_memory_size': 128,
                'conv3d_params':
                    {'inchan': 3,
                     'kernel_size': [5, 1, 1],
                     'outchan': 128,
                     'padding': [2, 0, 0]
                     },
                'hidsize': 4096,
                'img_shape': [128, 128, 128],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 16,
                'init_norm_kwargs':
                    {'batch_norm': False,
                     'group_norm_groups': 1
                     },
                'n_recurrence_layers': 2,
                'only_img_input': True,
                'pointwise_ratio': 4,
                'pointwise_use_activation': False,
                'recurrence_is_residual': True,
                'recurrence_type': 'transformer',
                'single_output': True,
                'timesteps': 128,
                'use_pointwise_layer': True,
                'use_pre_lstm_ln': False,
                'num_class': 4,
                'device': 'cuda',
                }
    args = argparse.Namespace(**net_args)
    model = Model(args)
    x = torch.randn(1, 128, 128, 128, 3).to('cuda')
    a = model(x)
    print(a.shape)
