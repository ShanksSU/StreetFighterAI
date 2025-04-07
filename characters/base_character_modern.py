# base_character_modern.py
import controls.control_keyboard_keys as kb
from controls.control_keyboard_keys import Direction, Button
import time
from enum import Enum
from functools import wraps

def button_action(tap_duration=0.05):
    """Decorator for simple button press actions"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            key = func(self, *args, **kwargs)
            kb.press_key(key)
            time.sleep(tap_duration)
            kb.release_key(key)
            return True
        return wrapper
    return decorator


class BaseCharacterModern:
    def __init__(self):
        # Movement states
        self.active_Key = set()
    
    # === Attack Commands ===
    @button_action()
    def light_attack(self):
        return Button.LIGHT.value
        
    @button_action()
    def medium_attack(self):
        return Button.MEDIUM.value

    @button_action()
    def heavy_attack(self):
        return Button.HEAVY.value

    @button_action()
    def special_attack(self):
        return Button.SPECIAL.value
    
    
    # === Basic Commands ===
    @button_action()
    def throw(self):
        return Button.THROW.value

    def drive_impact(self):
        kb.tap_key(Button.IMPACT.value)

    def assist(self):
        kb.tap_key(Button.ASSIST.value)

    @button_action(tap_duration=0.5)
    def drive_parry(self):
        return Button.PARRY.value

    # === Hold, Release and Stop Key ===
    def hold_key(self, Key):
        """Hold a direction key and track state"""
        kb.press_key(Key.value)
        self.active_Key.add(Key)
        
    def release_key(self, Key):
        """Release a direction key and update state"""
        kb.release_key(Key.value)
        if Key in self.active_Key:
            self.active_Key.remove(Key)
    
    def stop_movement(self):
        for Key in list(self.active_Key):
            self.release_key(Key)
    
    # Tap movement actions
    @button_action()
    def move_jump(self):
        return Direction.UP.value
        
    def move_left(self):
        kb.tap_key(Direction.LEFT.value)
        
    def move_crouch(self):
        kb.tap_key(Direction.DOWN.value)
        
    def move_right(self):
        kb.tap_key(Direction.RIGHT.value)
            
    def move_continuously(self, direction, duration=0.02):
        self.stop_movement()
        self.hold_key(direction)
        time.sleep(duration)
        
    def move_left_continuously(self, duration=0.02):
        self.move_continuously(Direction.LEFT, duration)
        
    def move_right_continuously(self, duration=0.02):
        self.move_continuously(Direction.RIGHT, duration)
        
    def move_crouch_continuously(self, duration=0.02):
        self.move_continuously(Direction.DOWN, duration)
    
    def down_left_continuously(self, duration=0.02):
        self.stop_movement()
        self.hold_key(Direction.DOWN)
        self.hold_key(Direction.LEFT)
        time.sleep(duration)
        
    def down_right_continuously(self, duration=0.02):
        self.stop_movement()
        self.hold_key(Direction.DOWN)
        self.hold_key(Direction.RIGHT)
        time.sleep(duration)
        
    def up_left_continuously(self, duration=0.02):
        self.stop_movement()
        self.hold_key(Direction.UP)
        self.hold_key(Direction.LEFT)
        time.sleep(duration)
        
    def up_right_continuously(self, duration=0.02):
        self.stop_movement()
        self.hold_key(Direction.UP)
        self.hold_key(Direction.RIGHT)
        time.sleep(duration)