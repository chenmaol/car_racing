import win32api
import win32con
import win32gui
import time
import ctypes
# pywin32

class Action():
    """
        Output action on keyboard and mouse
    """

    def __init__(self):
        self.mapVirtualKey = ctypes.windll.user32.MapVirtualKeyA
        self.map = {'w': 87, 's': 83, 'a': 65, 'd': 68, 'r': 82, 'esc': 27, 'enter': 13, 'ctrl': 17}
        # self.map = {'w': 38, 's': 40, 'a': 37, 'd': 39, 'r':82 ,'esc': 27, 'enter': 13}

    def down_key(self, value):
        win32api.keybd_event(self.map[value], self.mapVirtualKey(self.map[value], 0), 0, 0)

    def up_key(self, value):
        win32api.keybd_event(self.map[value], self.mapVirtualKey(self.map[value], 0), win32con.KEYEVENTF_KEYUP, 0)

    def press_key(self, value, internal=0.1):
        self.down_key(value)
        time.sleep(internal)
        self.up_key(value)
        time.sleep(0.01)

    def left_click(self, internal=0.5):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(internal)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.1)

    def move_mouse(self, pos):
        for i in range(5):
            win32api.SetCursorPos(pos)
            time.sleep(0.1)



if __name__ == '__main__':
    action = Action()
    action.down_key('w')