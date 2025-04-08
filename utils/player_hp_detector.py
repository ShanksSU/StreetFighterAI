import cv2
import math
import numpy as np

def calculate_hp_percentage(hp_part, color_ranges, grayscale_range):
    hsv_image = cv2.cvtColor(hp_part, cv2.COLOR_BGR2HSV)
    mask_gray = cv2.inRange(hsv_image, grayscale_range[0], grayscale_range[1])
    valid_area_mask = cv2.bitwise_not(mask_gray)
    valid_pixel_count = np.sum(valid_area_mask > 0)
    
    if valid_pixel_count == 0:
        return 0
    
    hp_pixel_count = 0
    for lower_color, upper_color in color_ranges:
        mask_color = cv2.inRange(hsv_image, lower_color, upper_color)
        mask_color_clean = cv2.bitwise_and(mask_color, valid_area_mask)
        hp_pixel_count += np.sum(mask_color_clean > 0)
    
    return math.ceil((hp_pixel_count / valid_pixel_count) * 100)

def L_player_HP(player_HP_img):
    color_ranges = [
        (np.array([130, 40, 80]), np.array([180, 255, 255])),  # Pink
        (np.array([25, 120, 230]), np.array([30, 160, 255]))   # Yellow
    ]
    grayscale_range = (np.array([0, 0, 180]), np.array([180, 50, 255]))
    L_player_HP_part = player_HP_img[:, 15:448] # (int(player_HP.shape[1] / 2) - 77) = 448
    return calculate_hp_percentage(L_player_HP_part, color_ranges, grayscale_range)

def R_player_HP(player_HP_img):
    color_ranges = [
        (np.array([100, 70, 50]), np.array([130, 255, 255])),  # Blue
        (np.array([25, 120, 230]), np.array([30, 160, 255]))   # Yellow
    ]
    grayscale_range = (np.array([90, 0, 180]), np.array([150, 50, 255]))
    R_player_HP_part = player_HP_img[:, 602:-15] # (int(player_HP_img.shape[1] / 2) + 77) = 602
    percentage = calculate_hp_percentage(R_player_HP_part, color_ranges, grayscale_range)
    return percentage + 1 if percentage > 0 else 0