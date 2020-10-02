from pynput import keyboard

def on_press(key):
        print('{0}'.format(key))

with keyboard.Listener(
        on_press=on_press
       ) as listener:
    listener.join()

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press)
listener.start()