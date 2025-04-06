import time
import keyboard

def pause_game(paused):
    if keyboard.is_pressed('t'):
        paused = not paused
        print('pause game' if paused else 'start game')
        time.sleep(1)

    while paused:
        print('paused')
        if keyboard.is_pressed('t'):
            paused = False
            print('start game')
            time.sleep(1)
            break
    return paused
