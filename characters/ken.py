from .base_character_classic import BaseCharacterClassic
from .base_character_modern import BaseCharacterModern
import controls.control_keyboard_keys as kb
import time

class Ken:
    def __init__(self, attack_mode='modern'):
        self.attack_mode = attack_mode.lower()
        self.impl = BaseCharacterModern() if attack_mode == 'modern' else BaseCharacterClassic()

    # Basic attack wrapper
    def light(self):
        if self.attack_mode == 'modern':
            self.impl.light_attack()
        else:
            self.impl.light_punch()

    def medium(self):
        if self.attack_mode == 'modern':
            self.impl.medium_attack()
        else:
            self.impl.medium_punch()

    def heavy(self):
        if self.attack_mode == 'modern':
            self.impl.heavy_attack()
        else:
            self.impl.heavy_punch()

    def hadouken(self):
        """Hadouken"""
        if self.attack_mode == 'modern':
            kb.tap_key('I')
        elif self.attack_mode == 'classic':
            print("Hadouken Classic")

    def shoryuken(self):
        """Shoryuken"""
        if self.attack_mode == 'modern':
            with kb.hold_keys(['D', 'I']):
                time.sleep(0.05)
            time.sleep(2)
        elif self.attack_mode == 'classic':
            print("Shoryuken Classic")

    def dragonlash_kick(self):
        """Dragonlash Kick"""
        if self.attack_mode == 'modern':
            with kb.hold_keys(['A', 'I']):
                time.sleep(0.05)
        elif self.attack_mode == 'classic':
            print("Dragonlash Kick Classic")
