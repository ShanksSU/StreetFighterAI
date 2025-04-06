import script.control_keyboard_keys as kb
import time

class BaseCharacterModern:
    # === Attack Commands ===
    def light_attack(self):     # Light Attack -> X
        kb.tap_key('U')

    def medium_attack(self):    # Medium Attack -> A
        kb.tap_key('J')

    def heavy_attack(self):     # Heavy Attack -> B
        kb.tap_key('K')

    def special_attack(self):   # Special Attack (SP) -> Y
        kb.tap_key('I')

    def throw(self):            # Throw -> LT
        kb.tap_key('H')

    def drive_impact(self):     # Drive Impact (Burst) -> LB
        kb.tap_key('Y')

    def assist(self):           # Assist -> RT
        kb.tap_key('L')

    def drive_parry(self):      # Drive Parry -> RB
        kb.tap_key('O')

    # === Basic Movement (Tap) ===
    def move_jump(self):        # Jump -> W
        kb.tap_key('W')

    def move_left(self):        # Move Left -> A
        kb.tap_key('A')

    def move_crouch(self):      # Crouch -> S
        kb.tap_key('S')

    def move_right(self):       # Move Right -> D
        kb.tap_key('D')

    # === Basic Movement (Hold/Release) ===
    def hold_jump(self):
        kb.press_key('W')

    def release_jump(self):
        kb.release_key('W')

    def hold_left(self):
        kb.press_key('A')

    def release_left(self):
        kb.release_key('A')

    def hold_crouch(self):
        kb.press_key('S')

    def release_crouch(self):
        kb.release_key('S')

    def hold_right(self):
        kb.press_key('D')

    def release_right(self):
        kb.release_key('D')
