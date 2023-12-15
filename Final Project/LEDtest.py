import RPi.GPIO as GPIO
import time

# do this for it to work: pip install RPi.GPIO

# Set the GPIO mode to BCM (Broadcom SOC channel)
GPIO.setmode(GPIO.BCM)

# Set up GPIO pin 26 as an output
led_pin = 26
GPIO.setup(led_pin, GPIO.OUT)

try:
    while True:
        # Turn on the LED
        GPIO.output(led_pin, GPIO.HIGH)
        print("LED ON")
        
        # Wait for 1 second
        time.sleep(1)

        # Turn off the LED
        GPIO.output(led_pin, GPIO.LOW)
        print("LED OFF")

        # Wait for 1 second
        time.sleep(1)

except KeyboardInterrupt:
    # If the user presses Ctrl+C, cleanup and exit
    GPIO.cleanup()
    print("Exiting script.")
