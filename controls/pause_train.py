# controls/pause_train.py
import time
import keyboard

def pause_train(paused):
    if keyboard.is_pressed('t'):
        paused = not paused
        print('Pause' if paused else 'Resume')
        time.sleep(1)  # Prevent key bouncing

    while paused:
        print('\rPaused... Press T to resume', end='')
        time.sleep(0.5)  # Reduce CPU usage while paused
        if keyboard.is_pressed('t'):
            paused = False
            print('\nResuming training')
            time.sleep(1)  # Prevent key bouncing
            break
    return paused