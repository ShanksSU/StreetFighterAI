# ken.py
from .base_character_classic import BaseCharacterClassic
from .base_character_modern import BaseCharacterModern, Direction, Button
import utils.control_keyboard_keys as kb
import time

class Ken:
    def __init__(self, attack_mode='modern'):
        self.attack_mode = attack_mode.lower()
        self.impl = BaseCharacterModern() if self.attack_mode == 'modern' else BaseCharacterClassic()
        self.active_Key = set()
    
    # === Hold, Release and Stop Key ===
    def hold_key(self, Key):
        kb.press_key(Key.value)
        self.active_Key.add(Key)
        
    def release_key(self, Key):
        kb.release_key(Key.value)
        if Key in self.active_Key:
            self.active_Key.remove(Key)
    
    def stop_movement(self):
        for Key in list(self.active_Key):
            self.release_key(Key)
    
    # === Skill ===
    def hadouken(self):
        self.impl.special_attack()

    def hadouken_OD(self):
        kb.press_key(Button.ASSIST.value)
        self.impl.special_attack()
        kb.release_key(Button.ASSIST.value)

    def shoryuken(self):
        self.stop_movement()
        self.hold_key(Direction.RIGHT)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()
        
    def shoryuken_OD(self):
        self.stop_movement()
        self.hold_key(Button.ASSIST)
        self.hold_key(Direction.RIGHT)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()

    def dragonlash_kick(self):
        self.stop_movement()
        self.hold_key(Direction.LEFT)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()
        
    def dragonlash_kick_OD(self):
        self.stop_movement()
        self.hold_key(Button.ASSIST)
        self.hold_key(Direction.LEFT)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()
        
    def jinrai_kick(self):
        self.stop_movement()
        self.hold_key(Direction.DOWN)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()

    def jinrai_kick_OD(self):
        self.stop_movement()
        self.hold_key(Button.ASSIST)
        self.hold_key(Direction.DOWN)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()

    def low_spinning_sweep(self):
        self.stop_movement()
        self.impl.down_right_continuously()
        self.hold_key(Button.HEAVY)
        time.sleep(0.05)
        self.stop_movement()

    def quick_dash(self):
        self.stop_movement()
        self.hold_key(Button.MEDIUM)
        self.hold_key(Button.HEAVY)
        time.sleep(0.05)
        self.stop_movement()
        
    def DOWN_SP(self):
        kb.press_key(Direction.DOWN.value)
        kb.press_key(Button.SPECIAL.value)
        time.sleep(0.05)
        kb.release_key(Direction.DOWN.value)
        kb.release_key(Button.SPECIAL.value)
        time.sleep(0.05)
        
    def DOWN_SP_OD(self):
        kb.press_key(Button.ASSIST.value)
        kb.press_key(Direction.DOWN.value)
        kb.press_key(Button.SPECIAL.value)
        time.sleep(0.05)
        kb.release_key(Direction.DOWN.value)
        kb.release_key(Button.SPECIAL.value)
        kb.release_key(Button.ASSIST.value)
        time.sleep(0.05)
        
    def tatsumaki_senpukyaku_OD(self):
        self.quick_dash()
        self.hold_key(Direction.DOWN)
        self.hold_key(Button.SPECIAL)
        time.sleep(0.05)
        self.stop_movement()
        
    def thunder_kick(self): # ?
        kb.press_key(Direction.DOWN.value)
        kb.press_key(Direction.LEFT.value)
        kb.press_key(Button.HEAVY.value)
        time.sleep(0.05)
        kb.release_key(Button.HEAVY.value)
        kb.release_key(Direction.LEFT.value)
        kb.release_key(Direction.DOWN.value)
        time.sleep(0.05)
        
    def jump_in_heavy(self):
        self.impl.move_jump()
        time.sleep(0.3)
        self.impl.heavy_attack()