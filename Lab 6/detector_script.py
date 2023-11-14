import os
import time
import board
import ssl

import paho.mqtt.client as mqtt
import uuid

global prev_message
prev_message = ''

client = mqtt.Client(str(uuid.uuid1()))
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.username_pw_set('idd', 'device@theFarm')

client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

topic = 'IDD/cool_table/spi'
KEYWORD = 'ben'
CONTROLLER = True

client.subscribe(topic)




def on_message(client, userdata, msg):
    global prev_message 
    message = msg.payload.decode('UTF-8')
    if message == prev_message:
        return
    os.system('cvlc --play-and-exit klaxon.mp3')
    os.system('echo "someone is talking about you! They said:" | festival --tts')
    f = open('tmp.txt', 'w+')
    f.write(message)
    f.close()
    os.system('festival --tts tmp.txt')
    prev_message = message
client.on_message = on_message
client.loop_forever()
