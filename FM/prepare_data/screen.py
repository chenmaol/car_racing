# import time
from time import time
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

'''
Important links for this script
https://learncodebygaming.com/blog/fast-window-capture
'''


class MyScreen():
    hwin = None
    top = left = right = bottom = width = height = -1

    def __init__(self, region=None, window_name='@#$_lorem_ipsum', title_size=0, border=0):
        # find the handle for the window we want to capture
        self.hwin = win32gui.FindWindow(None, window_name)
        if self.hwin:
            print("Found Capture window by name")
            window_rect = win32gui.GetWindowRect(self.hwin)
            self.left = window_rect[0]
            self.top = window_rect[1]
            self.right = window_rect[2]
            self.bottom = window_rect[3]
            self.width = self.right - self.left
            self.height = self.bottom - self.top

            # since using window handle so setting left and top to zero
            self.left = 0
            self.top = 0
        else:
            print("Using Window capture with region provided")
            self.hwin = win32gui.GetDesktopWindow()
            if region:
                self.left, self.top, self.right, self.bottom = region
                self.width = self.right - self.left
                self.height = self.bottom - self.top
            else:
                print("Using whole desktop capture")
                self.left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
                self.top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
                self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
                self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
                self.right = self.left + self.width
                self.bottom = self.top + self.height

        self.top += title_size

        self.left += border
        self.top += border
        self.width -= 2 * border
        self.height -= (border + title_size)

        assert (self.hwin)
        assert (self.left != -1)
        assert (self.top != -1)
        assert (self.right != -1)
        assert (self.bottom != -1)
        assert (self.width != -1)
        assert (self.height != -1)

    def get_frame(self):

        hwindc = win32gui.GetWindowDC(self.hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, self.width, self.height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (self.width, self.height), srcdc, (self.left, self.top), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.height, self.width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type()
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img


if __name__ == "__main__":
    # time.sleep(2)
    # myScreen = MyScreen(window_name="self_driving_car_nanodegree_program", border=20, title_size=40)
    # myScreen = MyScreen(window_name="self_driving_car_nanodegree_program", border=0, title_size=0)
    # myScreen = MyScreen(region=(860,110,1880,900))
    myScreen = MyScreen()
    loop_time = time()
    while (True):

        # get an updated image of the game
        frame = myScreen.get_frame()

        cv2.imshow('Computer Vision', frame)

        # debug the loop rate
        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
