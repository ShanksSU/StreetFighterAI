import utils.control_keyboard_keys as kb
import time

class BaseCharacterClassic:
    # Light Punch: Press 'U' key, X
    def light_punch(self):
        kb.tap_key('U')
        
    # Medium Punch: Press 'I' key, Y
    def medium_punch(self):
        kb.tap_key('I')
        
    # Heavy Punch: Press 'O' key, RB
    def heavy_punch(self):
        kb.tap_key('O')
        
    # Light Kick: Press 'J' key, A
    def light_kick(self):
        kb.tap_key('J')

    # Medium Kick: Press 'K' key, B
    def medium_kick(self):
        kb.tap_key('K')

    # Heavy Kick: Press 'L' key, RT
    def heavy_kick(self):
        kb.tap_key('L')

    # Parry: Medium Punch + Medium Kick, defensive action, LT
    def parry(self):        
        kb.tap_key('H')
        time.sleep(0.1)

    # Burst: Heavy Punch + Heavy Kick, offensive action, LB
    def burst(self):        
        kb.tap_key('Y')
        time.sleep(0.1)
