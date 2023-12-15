import time
from adafruit_servokit import ServoKit

# Create a servo kit object with 16 channels
kit = ServoKit(channels=16, frequency=60)

# Depending on your servo make, the pulse width min and max may vary,
# you want these to be as small/large as possible without hitting the hard stop
# for max range. You'll have to tweak them as necessary to match the servos you have!
SERVOMIN = 150  # this is the 'minimum' pulse length count (out of 4096)
SERVOMAX = 600  # this is the 'maximum' pulse length count (out of 4096)

# Servo numbers for each servo at angle 0
servo_numbers = [0, 1, 2, 3, 12, 13, 14] # 0, 1, 2, 3, 12, 13, 14

# Function to set the servo pulse length in seconds
def set_servo_pulse(n, pulse):
    pulselength = 1000000.0  # 1,000,000 us per second
    pulselength /= 60.0  # 60 Hz
    pulselength /= 4096.0  # 12 bits of resolution
    pulse *= 1000.0
    pulse /= pulselength
    kit.servo[n].angle = pulse


def move_servo(servo_num, start, end, step):
    for pulselen in range(start, end, step):
        # Normalize pulselen to the servo's angle range (0 to 180)
        normalized_angle = (pulselen - SERVOMIN) / (SERVOMAX - SERVOMIN) * 180

        # Reverse direction for servos 2, 13, and 14
        if servo_num in [2, 13, 14]:
            normalized_angle = 180 - normalized_angle

        kit.servo[servo_num].angle = normalized_angle
        time.sleep(0.001)
        # print(normalized_angle)
    


# Set all servos to angle 0
for servo_num in servo_numbers:
    normalized_angle = 0

    if servo_num in [2, 13, 14]:
        normalized_angle = 180 - normalized_angle
    
    if servo_num in [3]:
        normalized_angle = 90 - normalized_angle
    
    kit.servo[servo_num].angle = normalized_angle

# Main loop
"""
Servo num 0 is the left hand
Servo num 1 is the left elbow
Servo num 2 is the left shoulder

Servo num 3 is the head 

Servo num 14 is the right hand
Servo num 13 is the right elbow
Servo num 12 is the right shoulder

"""

while True:

    time.sleep(0.5)
    
    for servo_num in servo_numbers:
        
        move_servo(servo_num, SERVOMIN, SERVOMAX, 1)

        time.sleep(0.5)

        move_servo(servo_num, SERVOMAX, SERVOMIN, -1)

        time.sleep(0.5)

  
