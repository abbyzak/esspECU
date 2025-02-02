import machine
import time

BUTTON_PIN_ADD = 19  # Button for addition
BUTTON_PIN_SUB = 4  # Button for subtraction
BUTTON_PIN_ACCEPT = 5  # Button for accepting and terminating

button_add = machine.Pin(BUTTON_PIN_ADD, machine.Pin.IN, machine.Pin.PULL_UP)
button_sub = machine.Pin(BUTTON_PIN_SUB, machine.Pin.IN, machine.Pin.PULL_UP)
button_accept = machine.Pin(BUTTON_PIN_ACCEPT, machine.Pin.IN, machine.Pin.PULL_UP)

value = 0

def listen_buttons():
    global value
    while True:
        if button_add.value() == 0:
            value += 1
            print("Value:", value)
            time.sleep(0.5)  # Debounce delay
        
        if button_sub.value() == 0:
            value -= 1
            print("Value:", value)
            time.sleep(0.5)  # Debounce delay
        
        if button_accept.value() == 0:
            print("Accept")
            break
        
        time.sleep(0.1)

listen_buttons()

