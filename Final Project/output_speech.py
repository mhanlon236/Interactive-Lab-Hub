import os
import time
import board
import ssl

import paho.mqtt.client as mqtt
import uuid

import RPi.GPIO as GPIO
import time

# Set the GPIO mode to BCM (Broadcom SOC channel)
GPIO.setmode(GPIO.BCM)

# Set up GPIO pin 26 as an output
led_pin = 26
GPIO.setup(led_pin, GPIO.OUT)

global prev_message
prev_message = ''

client = mqtt.Client(str(uuid.uuid1()))
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.username_pw_set('idd', 'device@theFarm')

client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

topic = 'IDD/cool_table/robot'

client.subscribe(topic)


def on_message(client, userdata, msg):
    global prev_message 
    message = msg.payload.decode('UTF-8')
    if message == prev_message:
        return
    f = open('tmp.txt', 'w+')
    f.write(message)
    f.close()

    GPIO.output(led_pin, GPIO.HIGH)
    os.system('festival --tts tmp.txt')
    GPIO.output(led_pin, GPIO.LOW)

    prev_message = message
client.on_message = on_message

try:
    client.loop_forever()

except KeyboardInterrupt:
    # If Ctrl+C is pressed, turn off the LED and clean up
    GPIO.output(led_pin, GPIO.LOW)
    GPIO.cleanup()
    print("Script terminated by user. LED turned off and GPIO cleaned up.")


