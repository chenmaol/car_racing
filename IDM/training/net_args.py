network_args = {'attention_heads': 32,
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