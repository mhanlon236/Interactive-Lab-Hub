import paho.mqtt.client as mqtt
import uuid
import ssl
import digitalio
import board

from adafruit_rgb_display.rgb import color565
import adafruit_rgb_display.st7789 as st7789
from PIL import Image, ImageDraw, ImageFont

# The display uses a communication protocol called SPI.
# SPI will not be covered in depth in this course. 
# you can read more https://www.circuitbasics.com/basics-of-the-spi-communication-protocol/
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = None
BAUDRATE = 64000000  # the rate  the screen talks to the pi
# Create the ST7789 display:
display = st7789.ST7789(
    board.SPI(),
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=135,
    height=240,
    x_offset=53,
    y_offset=40,
)

height =  display.height
width = display.width 
image = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image)

# the # wildcard means we subscribe to all subtopics of IDD
topic = 'IDD/#'

# some other examples
# topic = 'IDD/a/fun/topic'

#this is the callback that gets called once we connect to the broker. 
#we should add our subscribe functions here as well
def on_connect(client, userdata, flags, rc):
	print(f"connected with result code {rc}")
	client.subscribe(topic)
	# you can subsribe to as many topics as you'd like
	# client.subscribe('some/other/topic')


# this is the callback that gets called each time a message is recived
def on_message(cleint, userdata, msg):
	# you can filter by topics
	# if msg.topic == 'IDD/some/other/topic': do thing
    if msg.topic == 'IDD/colors':
        print(f"topic: {msg.topic} msg: {msg.payload.decode('UTF-8')}")
        parts = msg.payload.decode('UTF-8').split(',')

        # Convert the strings to integers
        integers = [int(part) for part in parts]
        colors = list(map(int, msg.payload.decode('UTF-8').split(',')))
        a = colors[3]
        colors = tuple(map(lambda x: int(255*(1-(a/65536))*255*(x/65536)), colors))
        print(colors)
        # display.fill(color565(colors[0], colors[1], colors[2]))
        draw.rectangle((0, 0, width, height), fill=colors[:3])
        display.image(image)
        


# Every client needs a random ID
client = mqtt.Client(str(uuid.uuid1()))
# configure network encryption etc
client.tls_set(cert_reqs=ssl.CERT_NONE)
# this is the username and pw we have setup for the class
client.username_pw_set('idd', 'device@theFarm')

# attach out callbacks to the client
client.on_connect = on_connect
client.on_message = on_message

#connect to the broker
client.connect(
    'farlab.infosci.cornell.edu',
    port=8883)

# this is blocking. to see other ways of dealing with the loop
#  https://www.eclipse.org/paho/index.php?page=clients/python/docs/index.php#network-loop
client.loop_forever()