import ssl

ssl._create_default_https_context = ssl._create_unverified_context
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
from Model import model


class Agent():
    """
    Sample code to run the agent
    1. we need to intialize the inference module with model as the input
    including model checkpoints and forward function or we can just use model forward
    2. the game environment which takes an action and return next frame.
    3. The next frame can be used as current frame in a loop
    """

    def __init__(self, model, env):
        self.model = model
        self.model.eval()
        self.env = env
        


    def run_agent(self, cur_frame):
        while True:
            if not self.env.check_if_game_end(cur_frame):
                with torch.no_grad():
                    action = self.model(cur_frame)
                cur_frame = self.env.step(action)
            else:
                print("Game terminated!")
                cur_frame = self.env.reset_game()
            
            if "Q" in self.key_check():
                print("exiting")


    def transform(self, cv2_imgs):
        


    @staticmethod
    def key_check():
        keyList = ["\b"]
        for char in "TFGHXCMQqpPYUN":
            keyList.append(char)
        keys = []

        for key in keyList:
            # the ord() function returns an integer representing the Unicode character
            # chr() goes opposite way
            if wapi.GetAsyncKeyState(ord(key)):
                keys.append(key)
        # doesn't work to catch shift...
        return keys

class OCRModule:
    """
    recognize text number
    """

    def __init__(self):
        import easyocr
        self.model = easyocr.Reader(['en'])
        self.rect = [0.0130, 0.0500, 0.0573, 0.0750]
        self.rect_menu = [0.0328, 0.0130, 0.1496, 0.0547]
        self.img_size = (1920, 1080)  # w, h

    def get_distance_area(self, img, rect):
        x1 = int(rect[0] * self.img_size[0])
        x2 = int(rect[2] * self.img_size[0])
        y1 = int(rect[1] * self.img_size[1])
        y2 = int(rect[3] * self.img_size[1])
        return img[y1:y2, x1:x2, :]

    def get_white_digit(self, img, thresh=230):
        b, g, r = cv2.split(img)
        _, b = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY)
        _, g = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
        _, r = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(cv2.bitwise_and(b, g), r)
        return mask

    def get_dist(self, img):
        roi = self.get_distance_area(img, self.rect)
        thres = self.get_white_digit(roi)
        result = self.model.readtext(thres)
        if len(result):
            s = result[0][1].lower().replace('m', '').strip()
            if s.isnumeric():
                return int(s)
        return None

    def get_text(self, img):
        roi = self.get_distance_area(img, self.rect_menu)
        # cv2.imwrite('text.jpg', roi)
        # thres = self.get_white_digit(roi)
        result = self.model.readtext(roi)
        if len(result):
            s = result[0][1].rstrip()
            return s
        return None


class MatchingModule:
    """
    Match template for start/restart/out-of-edge icons, speed
    """

    def __init__(self):
        self.img_size = (1920, 1080)  # w, h
        self.edge_refer_img = cv2.imread('env_images/edge_refer.jpg')
        self.start_refer_img = cv2.imread('env_images/start_refer.jpg')
        self.restart_refer_img = cv2.imread('env_images/restart_refer.jpg')
        self.restart_text_refer_img = cv2.imread('env_images/restart_text_refer.jpg')
        self.continue_refer_img = cv2.imread('env_images/continue_refer.jpg')

        self.start_refer_img_binary = self.get_binary(self.start_refer_img)
        self.restart_refer_img_binary = self.get_binary(self.restart_refer_img)
        self.restart_text_refer_img_binary = self.get_binary(self.restart_text_refer_img)

        self.binary_thres = 220
        self.edge_thres = 0.25
        self.edge_rect = [0.8438, 0.1917, 0.8854, 0.2500]
        self.speed_rects = [
            [0.8713, 0.8850],
            [0.8870, 0.8850],
            [0.9036, 0.8850],
        ]
        self.speed_rect_size = [0.0130, 0.0400]
        # self.continue_rect = [0.0, 0.4456, 0.0328, 0.4973]
        self.continue_rect = [0.0, 0.3, 0.0328, 0.6]

    def get_distance_area(self, img, rect):
        x1 = int(rect[0] * self.img_size[0])
        x2 = int(rect[2] * self.img_size[0])
        y1 = int(rect[1] * self.img_size[1])
        y2 = int(rect[3] * self.img_size[1])
        return img[y1:y2, x1:x2, :]

    def get_binary(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thres = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        return thres

    def get_edge(self, img):
        roi = self.get_distance_area(img, rect=self.edge_rect)
        res = cv2.matchTemplate(roi, self.edge_refer_img, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return True if min_val < self.edge_thres else False

    def get_restart_text(self, img):
        res = cv2.matchTemplate(self.get_binary(img), self.restart_text_refer_img_binary, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        loc = [x + y // 2 for x, y in zip(min_loc, self.restart_text_refer_img.shape[:2][::-1])]
        return loc

    def get_restart(self, img):
        res = cv2.matchTemplate(self.get_binary(img), self.restart_refer_img_binary, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        loc = [x + y // 2 for x, y in zip(min_loc, self.restart_refer_img.shape[:2][::-1])]
        return loc

    def get_start(self, img):
        res = cv2.matchTemplate(self.get_binary(img), self.start_refer_img_binary, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        loc = [x + y // 2 for x, y in zip(min_loc, self.start_refer_img.shape[:2][::-1])]
        return loc, min_val

    def get_continue(self, img):
        h, w = img.shape[:2]
        x1 = int(w * self.continue_rect[0])
        y1 = int(h * self.continue_rect[1])
        x2 = int(w * self.continue_rect[2])
        y2 = int(h * self.continue_rect[3])
        res = cv2.matchTemplate(img[y1:y2, x1:x2, :], self.continue_refer_img, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return min_val

    def signal_logic(self, signals):
        if signals == [0, 0, 0, 0, 0, 0, 0]:
            return 0
        elif signals == [1, 1, 1, 1, 1, 1, 0]:
            return 0
        elif signals == [0, 0, 0, 0, 1, 1, 0]:
            return 1
        elif signals == [1, 0, 1, 1, 0, 1, 1]:
            return 2
        elif signals == [1, 0, 0, 1, 1, 1, 1]:
            return 3
        elif signals == [0, 1, 0, 0, 1, 1, 1]:
            return 4
        elif signals == [1, 1, 0, 1, 1, 0, 1]:
            return 5
        elif signals == [1, 1, 1, 1, 1, 0, 1]:
            return 6
        elif signals == [1, 1, 0, 0, 1, 1, 0]:
            return 7
        elif signals == [1, 1, 1, 1, 1, 1, 1]:
            return 8
        elif signals == [1, 1, 0, 1, 1, 1, 1]:
            return 9
        else:
            print('error signals:', signals)
            return None

    def single_number_logic(self, img, thres=200):
        h, w = img.shape[:2]
        area1 = img[:h // 8, w // 5:w * 4 // 5, :]
        area2 = img[h // 8:h * 2 // 5, w // 25:w // 5, :]
        area3 = img[h * 3 // 5: h * 7 // 8, w // 25:w // 5, :]
        area4 = img[h * 7 // 8:, w // 5:w * 4 // 5, :]
        area5 = img[h * 3 // 5: h * 7 // 8, w * 4 // 5:, :]
        area6 = img[h // 8:h * 2 // 5, w * 4 // 5:, :]
        area7 = img[h * 6 // 13: h * 7 // 13, w // 5:w * 4 // 5, :]
        # cv2.imwrite("im_digit.jpg", area1)

        signals = []
        for i, area in enumerate([area1, area2, area3, area4, area5, area6, area7]):
            # cv2.imwrite("im_digit_{}.jpg".format(i), area)
            # print(np.mean(area))
            signals.append(1 if np.mean(area) > thres else 0)
        num = self.signal_logic(signals)
        return num

    def get_speed(self, img):
        nums = ''
        for i, rect in enumerate(self.speed_rects):
            x1 = self.speed_rects[i][0]
            x2 = x1 + self.speed_rect_size[0]
            y1 = self.speed_rects[i][1]
            y2 = y1 + self.speed_rect_size[1]
            single_num_img = self.get_distance_area(img, [x1, y1, x2, y2])
            # cv2.imwrite("im_speed_{}.jpg".format(i), single_num_img)
            num = self.single_number_logic(single_num_img)
            # print(num)
            if num is None:
                cv2.imwrite('im_error_signals.jpg', single_num_img)
                return None
            else:
                nums += str(num)
        return int(nums)



class Env:
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 handle,
                 repeat_thres=10,
                 img_size=128,
                 stack_frames=1,
                 time_interval=0.5,
                 max_speed=30,
                 ):
        # init paras
        self.hwnd_target = handle
        self.action = Action()
        self.states = []
        self.action_spaces = ['w', 'a', 'd', '']
        self.size = (1920, 1080)
        self.ratio = 1
        self.repeat_nums = 0
        self.repeat_thres = repeat_thres
        self.stack_frames = stack_frames
        self.time_interval = time_interval
        self.max_speed = max_speed
        self.img_size = img_size

        # seg module
        # self.seg = SegModule()
        # matching module
        self.match = MatchingModule()
        # ocr module
        self.ocr = OCRModule()
        # map manager para
        self.menu_phase_check_point = [0.1500, 0.0513]  # x, y
        self.menu_phase_check_color = [[255, 255, 255], [0, 76, 255]]  # bgr
        # self.WALES_tracks = ['Wales shakedown', 'Hafren', 'Hafren reverse', 'Great Orme', 'Great Orme reverse', 'Brenig', 'Brenig reverse', 'Dyfi', 'Dyfi reverse']

        # back game
        win32gui.SetForegroundWindow(self.hwnd_target)
        time.sleep(1)
      

    def get_frame(self):
        w, h = self.size
        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        result = saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
        im_opencv.shape = (h, w, 4)
        im_opencv = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2BGR)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)
        return im_opencv

    def calc_reward(self, speed):
        return min(speed / self.max_speed, 1.0)

    def step(self, action):
        # take action
        t1 = (action[0] + 1) * self.time_interval / 2
        d = 1 if action[1] > 0 else 2
        t2 = abs(action[1]) * self.time_interval
        self.action.down_key(self.action_spaces[0])
        self.action.down_key(self.action_spaces[d])
        time.sleep(min(t1, t2))
        if t1 >= t2:
            self.action.up_key(self.action_spaces[d])
        else:
            self.action.up_key(self.action_spaces[0])
        time.sleep(max(t1, t2) - min(t1, t2))
        if t1 >= t2:
            self.action.up_key(self.action_spaces[0])
        else:
            self.action.up_key(self.action_spaces[d])
        time.sleep(self.time_interval - max(t1, t2))

        # get next state
        state = self.get_frame()
       
        return state

    def init_states(self):
        self.states = []
        state = self.get_frame()
        # preprocess frame
        # gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state_resize = cv2.resize(state, (self.img_size, self.img_size))

        for i in range(self.stack_frames):
            self.states.append(state_resize)
        self.repeat_nums = 0

    # ============== Control Functions ======================
    def init_run(self):
        for i in range(1):
            self.step([1, 0])
        time.sleep(5)

    def reset_car(self):
        self.action.press_key('r', internal=2)
        time.sleep(1)
        self.init_states()
        return np.stack(self.states, axis=0)

    def reset_game(self):
        self.pause_game()
        time.sleep(1)
        self.action.move_mouse([int(x * self.ratio) for x in self.match.get_restart(self.get_frame())])
        time.sleep(1)
        self.action.press_key('enter')
        time.sleep(1)
        self.action.move_mouse([int(x * self.ratio) for x in self.match.get_restart_text(self.get_frame())])
        time.sleep(1)
        self.action.press_key('enter')
        time.sleep(5)
        self.action.move_mouse([int(x * self.ratio) for x in self.match.get_start(self.get_frame())[0]])
        time.sleep(1)
        self.action.press_key('enter')
        time.sleep(1)
        self.init_states()
        self.init_run()
        self.init_states()
        return np.stack(self.states, axis=0)[-1]

    def pause_game(self):
        time.sleep(1)
        self.action.press_key('esc', internal=0.1)
        time.sleep(3)

    # ============== Map Switch Functions ======================
    def return_to_menu(self):
        max_try = 50
        cnt_try = 0
        suc = False
        while cnt_try < max_try:
            cnt_try += 1
            time.sleep(0.5)
            if self.check_if_return_to_menu():
                suc = True
                break
            self.action.press_key('enter')
        return suc

    def check_if_return_to_menu(self):
        frame = self.get_frame()
        text = self.ocr.get_text(frame)
        # print(text)
        if text == 'RALLY':
            return True
        return False

    def check_menu_phase(self):
        frame = self.get_frame()
        h, w = frame.shape[:2]
        y = int(h * self.menu_phase_check_point[1])
        x = int(w * self.menu_phase_check_point[0])
        check_point_color = frame[y, x, :]
        diff0 = sum([abs(a - b) for a, b in zip(check_point_color, self.menu_phase_check_color[0])])
        diff1 = sum([abs(a - b) for a, b in zip(check_point_color, self.menu_phase_check_color[1])])
        return 0 if diff0 < diff1 else 1

    def select_track(self, phase):
        if phase == 0:
            self.action.press_key('ctrl')
            self.action.press_key('enter')
        self.action.press_key('ctrl')
        self.action.press_key('enter')
        for i in range(80):
            self.action.press_key('enter')
            time.sleep(0.5)

    def switch_map(self):
        # return to menu
        if not self.return_to_menu():
            return
        # check menu phase
        phase = self.check_menu_phase()
        if phase == 1:
            self.action.press_key('esc')
        # select track
        self.select_track(phase)

    def check_if_game_end(self, frame):
        diff = self.match.get_continue(frame)
        # print(diff)
        if diff < 0.1:
            return True
        return False


if __name__ == '__main__':

    wrcg_env = Env(0x000D0400)
    first_frame = wrcg_env.get_frame()
    model = torch.load("a path to foundation model")
    agent = Agent(model=model, env=wrcg_env)
    agent.run_agent(first_frame)

    