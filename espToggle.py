import machine
import time

BUTTON_PIN = 5  # Instantaneous button
TOGGLE_BUTTON_PIN = 4  # Toggle button

button = machine.Pin(BUTTON_PIN, machine.Pin.IN, machine.Pin.PULL_UP)
toggle_button = machine.Pin(TOGGLE_BUTTON_PIN, machine.Pin.IN, machine.Pin.PULL_UP)

state = "Disabled"

def toggle_state():
    global state
    if state == "Disabled":
        state = "Engaged Mode"
    elif state == "Engaged Mode":
        state = "DisEngage Mode"
    elif state == "DisEngage Mode":
        state = "Engaged Mode"

while True:
    if toggle_button.value() == 0:
        toggle_state()
        print("State:", state)
        time.sleep(0.6)  # Debounce delay
    
    if state == "Engaged Mode":
        if button.value() == 0:
            print("State: Instantaneous Disabled")
            while button.value() == 0:
                time.sleep(0.1)  # Wait for button release
            print("State: Engaged Mode")
    
    time.sleep(0.1)

