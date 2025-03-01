# Chatterboxes
**Kenneth Lee, Gilberto Ruiz, Ben Setel, Gloria Hu, Yifan Yu, Michael Hanlon**
[![Watch the video](https://user-images.githubusercontent.com/1128669/135009222-111fe522-e6ba-46ad-b6dc-d1633d21129c.png)](https://www.youtube.com/embed/Q8FWzLMobx0?start=19)

In this lab, we want you to design interaction with a speech-enabled device--something that listens and talks to you. This device can do anything *but* control lights (since we already did that in Lab 1).  First, we want you first to storyboard what you imagine the conversational interaction to be like. Then, you will use wizarding techniques to elicit examples of what people might say, ask, or respond.  We then want you to use the examples collected from at least two other people to inform the redesign of the device.

We will focus on **audio** as the main modality for interaction to start; these general techniques can be extended to **video**, **haptics** or other interactive mechanisms in the second part of the Lab.

## Prep for Part 1: Get the Latest Content and Pick up Additional Parts 

### Pick up Web Camera If You Don't Have One

Students who have not already received a web camera will receive their [IMISES web cameras](https://www.amazon.com/Microphone-Speaker-Balance-Conference-Streaming/dp/B0B7B7SYSY/ref=sr_1_3?keywords=webcam%2Bwith%2Bmicrophone%2Band%2Bspeaker&qid=1663090960&s=electronics&sprefix=webcam%2Bwith%2Bmicrophone%2Band%2Bsp%2Celectronics%2C123&sr=1-3&th=1) on Thursday at the beginning of lab. If you cannot make it to class on Thursday, please contact the TAs to ensure you get your web camera. 

**Please note:** connect the webcam/speaker/microphone while the pi is *off*. 

### Get the Latest Content

As always, pull updates from the class Interactive-Lab-Hub to both your Pi and your own GitHub repo. There are 2 ways you can do so:

**\[recommended\]**Option 1: On the Pi, `cd` to your `Interactive-Lab-Hub`, pull the updates from upstream (class lab-hub) and push the updates back to your own GitHub repo. You will need the *personal access token* for this.

```
pi@ixe00:~$ cd Interactive-Lab-Hub
pi@ixe00:~/Interactive-Lab-Hub $ git pull upstream Fall2022
pi@ixe00:~/Interactive-Lab-Hub $ git add .
pi@ixe00:~/Interactive-Lab-Hub $ git commit -m "get lab3 updates"
pi@ixe00:~/Interactive-Lab-Hub $ git push
```

Option 2: On your your own GitHub repo, [create pull request](https://github.com/FAR-Lab/Developing-and-Designing-Interactive-Devices/blob/2022Fall/readings/Submitting%20Labs.md) to get updates from the class Interactive-Lab-Hub. After you have latest updates online, go on your Pi, `cd` to your `Interactive-Lab-Hub` and use `git pull` to get updates from your own GitHub repo.

## Part 1.
### Setup 

*DO NOT* forget to work on your virtual environment! 

Run the setup script
```chmod u+x setup.sh && sudo ./setup.sh  ```

### Text to Speech 

In this part of lab, we are going to start peeking into the world of audio on your Pi! 

We will be using the microphone and speaker on your webcamera. In the directory is a folder called `speech-scripts` containing several shell scripts. `cd` to the folder and list out all the files by `ls`:

```
pi@ixe00:~/speech-scripts $ ls
Download        festival_demo.sh  GoogleTTS_demo.sh  pico2text_demo.sh
espeak_demo.sh  flite_demo.sh     lookdave.wav
```

You can run these shell files `.sh` by typing `./filename`, for example, typing `./espeak_demo.sh` and see what happens. Take some time to look at each script and see how it works. You can see a script by typing `cat filename`. For instance:

```
pi@ixe00:~/speech-scripts $ cat festival_demo.sh 
#from: https://elinux.org/RPi_Text_to_Speech_(Speech_Synthesis)#Festival_Text_to_Speech
```
You can test the commands by running
```
echo "Just what do you think you're doing, Dave?" | festival --tts
```

Now, you might wonder what exactly is a `.sh` file? 
Typically, a `.sh` file is a shell script which you can execute in a terminal. The example files we offer here are for you to figure out the ways to play with audio on your Pi!

You can also play audio files directly with `aplay filename`. Try typing `aplay lookdave.wav`.

\*\***Write your own shell file to use your favorite of these TTS engines to have your Pi greet you by name.**\*\*
The shell file is called "greeting.sh"

---
Bonus:
[Piper](https://github.com/rhasspy/piper) is another fast neural based text to speech package for raspberry pi which can be installed easily through python with:
```
pip install piper-tts
```
and used from the command line. Running the command below the first time will download the model, concurrent runs will be faster. 
```
echo 'Welcome to the world of speech synthesis!' | piper \
  --model en_US-lessac-medium \
  --output_file welcome.wav
```
Check the file that was created by running `aplay welcome.wav`. Many more languages are supported and audio can be streamed dirctly to an audio output, rather than into an file by:

```
echo 'This sentence is spoken first. This sentence is synthesized while the first sentence is spoken.' | \
  piper --model en_US-lessac-medium --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -
```
  
### Speech to Text

Next setup speech to text. We are using a speech recognition engine, [Vosk](https://alphacephei.com/vosk/), which is made by researchers at Carnegie Mellon University. Vosk is amazing because it is an offline speech recognition engine; that is, all the processing for the speech recognition is happening onboard the Raspberry Pi. 
```
pip install vosk
pip install sounddevice
```

Test if vosk works by transcribing text:

```
vosk-transcriber -i recorded_mono.wav -o test.txt
```

You can use vosk with the microphone by running 
```
python test_microphone.py -m en
```

\*\***Write your own shell file that verbally asks for a numerical based input (such as a phone number, zipcode, number of pets, etc) and records the answer the respondent provides.**\*\*
This part was very difficult for me to figure out because when first created the sh file that uses ">" to print the "test_microphone.py" output to a text file. However, for some reason this causes the python file to not print out what the user says in the terminal anymore, so the text file stays empty. Instead, I had to first pipe the output to another python file called "handle_test_microphone.py" and then use that to output the result to the "numerical_input.txt" text file.

Shell file: numerical_input.sh
Helper python file: handle_test_microphone.py
Output text file: numerical_input.txt


### Serving Pages

In Lab 1, we served a webpage with flask. In this lab, you may find it useful to serve a webpage for the controller on a remote device. Here is a simple example of a webserver.

```
pi@ixe00:~/Interactive-Lab-Hub/Lab 3 $ python server.py
 * Serving Flask app "server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 162-573-883
```
From a remote browser on the same network, check to make sure your webserver is working by going to `http://<YourPiIPAddress>:5000`. You should be able to see "Hello World" on the webpage.

### Storyboard

Storyboard and/or use a Verplank diagram to design a speech-enabled device. (Stuck? Make a device that talks for dogs. If that is too stupid, find an application that is better than that.) 

\*\***Post your storyboard and diagram here.**\*\*

![1](https://github.com/Ruiznogueras05CT/Interactive-Lab-Hub/assets/142849822/e0e50998-942e-4427-bf0d-c26df9358c00)
![2](https://github.com/Ruiznogueras05CT/Interactive-Lab-Hub/assets/142849822/2bbb2807-09bc-4dbe-82fd-1c60b62831eb)
![3](https://github.com/Ruiznogueras05CT/Interactive-Lab-Hub/assets/142849822/a62e9ada-f8e4-4a7e-bee5-9685c92c93f3)
![4](https://github.com/Ruiznogueras05CT/Interactive-Lab-Hub/assets/142849822/05e0b538-0522-4435-bea4-06b67fcca294)


Write out what you imagine the dialogue to be. Use cards, post-its, or whatever method helps you develop alternatives or group responses. 

\*\***Please describe and document your process.**\*\*

# The Script:

Gil: Todavía estás trabajando en tu tarea? 

Random Person: (probably) huh? 

Device: Are you still working on your hw? 

Random Person: whatever they say

Gil: Buena suerte en tu tarea, espero que la termines pronto!

Device: Good luck on your hw, I hope you finish it soon!

Random Person: Thank you! 

End Scene


### Acting out the dialogue

Find a partner, and *without sharing the script with your partner* try out the dialogue you've designed, where you (as the device designer) act as the device you are designing.  Please record this interaction (for example, using Zoom's record feature).

\*\***Describe if the dialogue seemed different than what you imagined when it was acted out, and how.**\*\*

### The dialogue seemed different than what we imagined when it was acted out due to the fact that we didn't know at what time should the device react and translate the phrase. 

# The Video:

https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/0ced3de5-02f3-4889-9c5f-34baffc9b386

### Wizarding with the Pi (optional)
In the [demo directory](./demo), you will find an example Wizard of Oz project. In that project, you can see how audio and sensor data is streamed from the Pi to a wizard controller that runs in the browser.  You may use this demo code as a template. By running the `app.py` script, you can see how audio and sensor data (Adafruit MPU-6050 6-DoF Accel and Gyro Sensor) is streamed from the Pi to a wizard controller that runs in the browser `http://<YouPiIPAddress>:5000`. You can control what the system says from the controller as well!

\*\***Describe if the dialogue seemed different than what you imagined, or when acted out, when it was wizarded, and how.**\*\*

# Lab 3 Part 2

For Part 2, you will redesign the interaction with the speech-enabled device using the data collected, as well as feedback from part 1.

## Prep for Part 2

1. What are concrete things that could use improvement in the design of your device? For example: wording, timing, anticipation of misunderstandings...
2. What are other modes of interaction _beyond speech_ that you might also use to clarify how to interact?
3. Make a new storyboard, diagram and/or script based on these reflections.

## Prototype your system

The system should:
* use the Raspberry Pi 
* use one or more sensors
* require participants to speak to it. 

*Document how the system works*

Using festival, the device speaks to the user, asking them to say its activation phase. Using vosk, it then listens for this phrase. Once it hears the phrase, it launches the "color_changer" program. This time it asks the user to say a color. When the user says a color, the program changes the screen to that color. If the user says "party", it cycles colors and plays a song. 

![alt text](https://github.com/bensetel/Interactive-Lab-Hub/blob/Fall2023/Lab%203/Lab%203%20Part%20Two.png)
![alt text](https://github.com/bensetel/Interactive-Lab-Hub/blob/Fall2023/Lab%203/Lab%203%20Part%20Two%20(2).png)


*Include videos or screencaptures of both the system and the controller.*  

https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/38ef1163-de09-4bc0-827f-04d8e8955a1c


## Test the system
Try to get at least two people to interact with your system. (Ideally, you would inform them that there is a wizard _after_ the interaction, but we recognize that can be hard.)

Answer the following:

Question: What worked well about the system and what didn't?

Answer:
***The system exhibited proficiency in its color-changing mechanism, demonstrating seamless transitions. Additionally, the text-to-speech component effectively prompted users for their input. However, the speech-to-text functionality displayed significant limitations. Specifically, it often encountered challenges in accurately recognizing and interpreting spoken words.*** 

Question: What worked well about the controller and what didn't?

Answer:
***The controller's programming successfully rendered colors on the display, ensuring a vibrant user experience. However, the earlier mentioned deficiencies in the speech-to-text component extended to the controller, making the system's overall control less intuitive and at times challenging.*** 

Question: What lessons can you take away from the WoZ interactions for designing a more autonomous version of the system?

Answer:
***Drawing from the Wizard of Oz (WoZ) interactions, there are several insights that can guide the design of a more autonomous iteration. One possibility is leveraging the video input from the camera to gauge ambient light conditions, subsequently adjusting screen colors for optimal visibility. Furthermore, the system could be designed to autonomously cycle through an array of colors or even utilize the text-to-speech output to generate corresponding speech-to-text inputs, essentially enabling the system to "converse" with itself.*** 

Question: How could you use your system to create a dataset of interaction? What other sensing modalities would make sense to capture?

Answer:
***To curate a dataset of interactions, the system can be programmed to record user engagements. The incorporation of video and audio capturing capabilities, particularly through a webcam, would be instrumental in this endeavor. Additionally, the stored speech-to-text inputs can be systematically analyzed, leading to the creation of a comprehensive database spotlighting the most frequently used keywords. Beyond these, integrating video analytics can offer richer data. Another avenue worth exploring is the introduction of self-generated auditory cues, which could further enrich the interaction dataset.*** 
