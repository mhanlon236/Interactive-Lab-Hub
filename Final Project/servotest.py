import time
from adafruit_servokit import ServoKit

# Create a servo kit object with 16 channels
kit = ServoKit(channels=16, frequency=60)

# Depending on your servo make, the pulse width min and max may vary,
# you want these to be as small/large as possible without hitting the hard stop
# for max range. You'll have to tweak them as necessary to match the servos you have!
SERVOMIN = 150  # this is the 'minimum' pulse length count (out of 4096)
SERVOMAX = 600  # this is the 'maximum' pulse length count (out of 4096)

# Set the PWM frequency to 60 Hz (Analog servos run at ~60 Hz updates)


# Function to set the servo pulse length in seconds
def set_servo_pulse(n, pulse):
    pulselength = 1000000.0  # 1,000,000 us per second
    pulselength /= 60.0  # 60 Hz
    pulselength /= 4096.0  # 12 bits of resolution
    pulse *= 1000.0
    pulse /= pulselength
    kit.servo[n].angle = pulse


# Main loop
"""
Servo num 0 is the left hand, 0.4 is the zero state angle
Servo num 1 is the left elbow, 
Servo num 2 is the left shoulder

Servo num 14 is the right hand
Servo num 13 is the right elbow
Servo num 12 is the right shoulder

"""
servo_num = 1

while True:
    # print(servo_num)
    for pulselen in range(SERVOMIN, SERVOMAX):
        # Normalize pulselen to the servo's angle range (0 to 180)
        normalized_angle = (pulselen - SERVOMIN) / (SERVOMAX - SERVOMIN) * 180
        kit.servo[servo_num].angle = normalized_angle
        time.sleep(0.001)
    print(normalized_angle)

    time.sleep(0.5)

    for pulselen in range(SERVOMAX, SERVOMIN, -1):
        # Normalize pulselen to the servo's angle range (0 to 180)
        normalized_angle = (pulselen - SERVOMIN) / (SERVOMAX - SERVOMIN) * 180
        kit.servo[servo_num].angle = normalized_angle
        time.sleep(0.001)
    print(normalized_angle)

    time.sleep(0.5)

    # servo_num += 1
    if servo_num >= 16:
        servo_num = 0



