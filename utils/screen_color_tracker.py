import cv2
import keyboard
import numpy as np
from pynput.mouse import Listener
from PIL import ImageGrab

class MousePositionListener:
    def __init__(self):
        self.mouse_position = (0, 0)
        self._start_listener()

    def _start_listener(self):
        listener = Listener(on_move=self._on_move)
        listener.daemon = True
        listener.start()

    def _on_move(self, x, y):
        self.mouse_position = (x, y)

    def get_position(self):
        return self.mouse_position

class ScreenCapture:
    @staticmethod
    def grab_region(bbox):
        img = ImageGrab.grab(bbox)
        return np.array(img)

class ColorExtractor:
    @staticmethod
    def get_center_color(img, region_size):
        center = region_size
        return img[center, center]

    @staticmethod
    def get_rgb_color(bbox, region_size):
        screen_img = ScreenCapture.grab_region(bbox)
        return ColorExtractor.get_center_color(screen_img, region_size)

    @staticmethod
    def get_hsv_color(bbox, region_size):
        screen_img = ScreenCapture.grab_region(bbox)
        hsv_image = cv2.cvtColor(screen_img, cv2.COLOR_RGB2HSV)
        return ColorExtractor.get_center_color(hsv_image, region_size)

class MouseColorTracker:
    def __init__(self, region_size=10):
        self.mouse_listener = MousePositionListener()
        self.region_size = region_size

    def get_bbox(self):
        x, y = self.mouse_listener.get_position()
        offset = self.region_size
        return (x - offset, y - offset, x + offset, y + offset)

    def get_color_rgb(self):
        bbox = self.get_bbox()
        rgb_color = ColorExtractor.get_rgb_color(bbox, self.region_size)
        print(f"\rMouse Position: {self.mouse_listener.get_position()}, Color (RGB): {rgb_color}", end="")
        return rgb_color

    def get_color_hsv(self):
        bbox = self.get_bbox()
        hsv_color = ColorExtractor.get_hsv_color(bbox, self.region_size)
        print(f"\rMouse Position: {self.mouse_listener.get_position()}, Color (HSV): {hsv_color}", end="")
        return hsv_color
