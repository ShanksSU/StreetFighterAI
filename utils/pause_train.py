# controls/pause_train.py
import time
import keyboard

def pause_train(paused, env=None):
    if keyboard.is_pressed('t'):
        paused = not paused
        print('Pause' if paused else 'Resume')

        if paused and env:
            env.character.impl.stop_movement()

        time.sleep(1)
        
    while paused:
        print('\rPaused... Press T to resume', end='')
        time.sleep(0.5)

        if env:
            env.character.impl.stop_movement()
        
        if keyboard.is_pressed('t'):
            paused = False
            print('\nResuming training')
            time.sleep(1)
            break
    return paused