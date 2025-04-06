import cv2
import keyboard
import numpy as np
from pynput.mouse import Listener
from PIL import ImageGrab

class MouseColorTracker:
    def __init__(self, region_size=10):
        """
        Initializes the mouse color tracker.
        :param region_size: The side length of the capture region (in pixels).
        """
        self.mouse_position = (0, 0)
        self.region_size = region_size
        self._start_listener()

    def _start_listener(self):
        """Starts the mouse position listener."""
        listener = Listener(on_move=self._on_move)
        listener.daemon = True  # Set to run in the background
        listener.start()

    def _on_move(self, x, y):
        """Updates the mouse position."""
        self.mouse_position = (x, y)

    def _grab_screen(self, bbox):
        """Grabs a screen region and returns it as a numpy array (RGB format)."""
        img = ImageGrab.grab(bbox)
        return np.array(img)

    def _get_center_color(self, img):
        """Gets the color value of the center pixel of the image."""
        center = self.region_size
        return img[center, center]

    def get_color_rgb(self):
        """Gets the RGB color of the center of the mouse's current region."""
        x, y = self.mouse_position
        offset = self.region_size
        bbox = (x - offset, y - offset, x + offset, y + offset)
        screen_img = self._grab_screen(bbox)
        center_color = self._get_center_color(screen_img)
        print(f"\rMouse Position: {self.mouse_position}, Color (RGB): {center_color}", end="")
        return center_color

    def get_color_hsv(self):
        """Gets the HSV color of the center of the mouse's current region."""
        x, y = self.mouse_position
        offset = self.region_size
        bbox = (x - offset, y - offset, x + offset, y + offset)
        screen_img = self._grab_screen(bbox)
        hsv_image = cv2.cvtColor(screen_img, cv2.COLOR_RGB2HSV)
        center_hsv = self._get_center_color(hsv_image)
        print(f"\rMouse Position: {self.mouse_position}, Color (HSV): {center_hsv}", end="")
        return center_hsv
    
if __name__ == "__main__":
    tracker = MouseColorTracker(region_size=10)
    
    print("Press q to stop tracking.")
    try:
        while True:
            tracker.get_color_rgb()
            # tracker.get_color_hsv()
            cv2.waitKey(100)
            if keyboard.is_pressed('q'):
                break
    except KeyboardInterrupt:
        print("\nTracking ended.")
