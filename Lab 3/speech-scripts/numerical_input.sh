#https://elinux.org/RPi_Text_to_Speech_(Speech_Synthesis)

#!/bin/bash
say() { local IFS=+;/usr/bin/mplayer -ao alsa -really-quiet -noconsolecontrols "http://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&q=$*&tl=en"; }
#say $*
say "Hey beautiful, what is your phone number?"

python -u test_microphone.py -m en | python handle_test_microphone.py > numerical_input.txt