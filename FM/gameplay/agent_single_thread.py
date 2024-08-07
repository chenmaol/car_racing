import time
import ctypes
import win32api
import win32ui
import win32gui
import win32con
import numpy as np
import cv2
from action import Action
import torch
from model import Model
from torchvision.transforms import transforms
from screen import MyScreen
from collections import OrderedDict
from PIL import Image

class Agent:
    """
    Sample code to run the agent
    1. we need to intialize the inference module with model as the input
    including model checkpoints and forward function or we can just use model forward
    2. the game environment which takes an action and return next frame.
    3. The next frame can be used as current frame in a loop
    """

    def __init__(self, model_config_path, checkpoint_path, gap_len, env):
        # load model and env
        self.env = env
        self.model = Model(model_config_path, batch_size=1).to("cuda")
        self.gap_len = gap_len
        self.device = 'cuda'
        state_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 "module." 前缀
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.base_transform = transforms.Compose([
            transforms.Resize((self.env.img_size, self.env.img_size)),
            transforms.ToTensor(),
        ])

        self.model.reset_queue()

    def run_agent(self):
        while True:
            t0 = time.time()
            # get frame
            img = self.env.get_frame()
            img = self.base_transform(img).unsqueeze(0).unsqueeze(0).to(self.device).permute(0, 1, 3, 4, 2)
            # infer
            self.model.extract_cnn_feature(img)
            if len(self.model.queue_features) == 32:
                output = self.model.transformer_forward()
                action = torch.max(output, dim=-1)[1].cpu().squeeze().numpy()
                t1 = time.time()
                print(action[0], f'inference time:{t1 - t0}')
                # do action
                self.env.step(action)
                t2 = time.time()
                print(f'action time:{t2 - t1}')
            if "Q" in self.key_check():
                print("exiting")
                break

    @staticmethod
    def key_check():
        keyList = ["\b"]
        for char in "TFGHXCMQqpPYUN":
            keyList.append(char)
        keys = []

        for key in keyList:
            # the ord() function returns an integer representing the Unicode character
            # chr() goes opposite way
            if win32api.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys

class Env:
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 img_size=128,
                 time_interval=0.1,
                 ):
        # init paras
        self.action = Action()
        self.states = []
        self.myScreen = MyScreen((0, 0, 1280, 800))
        self.repeat_nums = 0
        self.time_interval = time_interval
        self.img_size = img_size
        self.action_spaces = ['w', 's', 'a', 'd']

        # back game
        # win32gui.SetForegroundWindow(self.hwnd_target)
        time.sleep(1)

    def get_frame(self):
        frame = self.myScreen.get_frame()
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame

    def step(self, action):
        start_time = time.time()

        # take action
        for i in range(4):
            if action[0, i] == 1:
                self.action.down_key(self.action_spaces[i])
        while time.time() < start_time + self.time_interval:
            pass
        for i in range(4):
            if action[0, i] == 1:
                self.action.up_key(self.action_spaces[i])


    # ============== Control Functions ======================

    def reset_car(self):
        self.action.press_key('r', internal=2)
        time.sleep(1)

    def pause_game(self):
        time.sleep(1)
        self.action.press_key('esc', internal=0.1)
        time.sleep(3)


if __name__ == '__main__':
    wrcg_env = Env()
    agent = Agent(model_config_path='model_config.yaml', checkpoint_path='best_model_2.pth', gap_len=0, env=wrcg_env)
    agent.run_agent()
