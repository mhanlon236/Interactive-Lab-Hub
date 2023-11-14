# Little Interactions Everywhere

**NAMES OF COLLABORATORS HERE**

Ben Setel, Kenneth Lee, Yifan Yu, Gloria Hu, Michael Hanlon

## Prep

1. Pull the new changes from the class interactive-lab-hub. (You should be familiar with this already!)
2. Install [MQTT Explorer](http://mqtt-explorer.com/) on your laptop. If you are using Mac, MQTT Explorer only works when installed from the [App Store](https://apps.apple.com/app/apple-store/id1455214828).
3. Readings before class:
   * [MQTT](#MQTT)
   * [The Presence Table](https://dl.acm.org/doi/10.1145/1935701.1935800) and [video](https://vimeo.com/15932020)


## Overview

The point of this lab is to introduce you to distributed interaction. We have included some Natural Language Processing (NLP) and Generation (NLG) but those are not really the emphasis. Feel free to dig into the examples and play around the code which you can integrate into your projects if wanted. However, we want to emphasize that the grading will focus on your ability to develop interesting uses for messaging across distributed devices. Here are the four sections of the lab activity:

A) [MQTT](#part-a)

B) [Send and Receive on your Pi](#part-b)

C) [Streaming a Sensor](#part-c)

D) [The One True ColorNet](#part-d)

E) [Make It Your Own](#part-)

## Part 1.

### Part A
### MQTT

MQTT is a lightweight messaging portal invented in 1999 for low bandwidth networks. It was later adopted as a defacto standard for a variety of [Internet of Things (IoT)](https://en.wikipedia.org/wiki/Internet_of_things) devices. 

#### The Bits

* **Broker** - The central server node that receives all messages and sends them out to the interested clients. Our broker is hosted on the far lab server (Thanks David!) at `farlab.infosci.cornell.edu/8883`. Imagine that the Broker is the messaging center!
* **Client** - A device that subscribes or publishes information to/on the network.
* **Topic** - The location data gets published to. These are *hierarchical with subtopics*. For example, If you were making a network of IoT smart bulbs this might look like `home/livingroom/sidelamp/light_status` and `home/livingroom/sidelamp/voltage`. With this setup, the info/updates of the sidelamp's `light_status` and `voltage` will be store in the subtopics. Because we use this broker for a variety of projects you have access to read, write and create subtopics of `IDD`. This means `IDD/ilan/is/a/goof` is a valid topic you can send data messages to.
*  **Subscribe** - This is a way of telling the client to pay attention to messages the broker sends out on the topic. You can subscribe to a specific topic or subtopics. You can also unsubscribe. Following the previouse example of home IoT smart bulbs, subscribing to `home/livingroom/sidelamp/#` would give you message updates to both the light_status and the voltage.
* **Publish** - This is a way of sending messages to a topic. Again, with the previouse example, you can set up your IoT smart bulbs to publish info/updates to the topic or subtopic. Also, note that you can publish to topics you do not subscribe to. 


**Important note:** With the broker we set up for the class, you are limited to subtopics of `IDD`. That is, to publish or subcribe, the topics will start with `IDD/`. Also, setting up a broker is not much work, but for the purposes of this class, you should all use the broker we have set up for you!


#### Useful Tooling

Debugging and visualizing what's happening on your MQTT broker can be helpful. We like [MQTT Explorer](http://mqtt-explorer.com/). You can connect by putting in the settings from the image below.


![input settings](imgs/mqtt_explorer.png?raw=true)


Once connected, you should be able to see all the messages under the IDD topic. , go to the **Publish** tab and try publish something! From the interface you can send and plot messages as well. Remember, you are limited to subtopics of `IDD`. That is, to publish or subcribe, the topics will start with `IDD/`.


<img width="1026" alt="Screen Shot 2022-10-30 at 10 40 32 AM" src="https://user-images.githubusercontent.com/24699361/198885090-356f4af0-4706-4fb1-870f-41c15e030aba.png">



### Part B
### Send and Receive on your Pi

[sender.py](./sender.py) and and [reader.py](./reader.py) show you the basics of using the mqtt in python. Let's spend a few minutes running these and seeing how messages are transferred and shown up. Before working on your Pi, keep the connection of `farlab.infosci.cornell.edu/8883` with MQTT Explorer running on your laptop.

**Running Examples on Pi**

* Install the packages from `requirements.txt` under a virtual environment:

  ```
  pi@raspberrypi:~/Interactive-Lab-Hub $ source circuitpython/bin/activate
  (circuitpython) pi@raspberrypi:~/Interactive-Lab-Hub $ cd Lab\ 6
  (circuitpython) pi@raspberrypi:~/Interactive-Lab-Hub/Lab 6 $ pip install -r requirements.txt
  ...
  ```
* Run `sender.py`, fill in a topic name (should start with `IDD/`), then start sending messages. You should be able to see them on MQTT Explorer.

  ```
  (circuitpython) pi@raspberrypi:~/Interactive-Lab-Hub/Lab 6 $ python sender.py
  >> topic: IDD/AlexandraTesting
  now writing to topic IDD/AlexandraTesting
  type new-topic to swich topics
  >> message: testtesttest
  ...
  ```
* Run `reader.py`, and you should see any messages being published to `IDD/` subtopics. Type a message inside MQTT explorer and see if you can receive it with `reader.py`.

  ```
  (circuitpython) pi@raspberrypi:~ Interactive-Lab-Hub/Lab 6 $ python reader.py
  ...
  ```

<img width="890" alt="Screen Shot 2022-10-30 at 10 47 52 AM" src="https://user-images.githubusercontent.com/24699361/198885135-a1d38d17-a78f-4bb2-91c7-17d014c3a0bd.png">



**\*\*\*Consider how you might use this messaging system on interactive devices, and draw/write down 5 ideas here.\*\*\***
![Security System](https://github.com/gloriahu28/Interactive-Lab-Hub/assets/142931503/af299139-65e8-46a0-a45f-2bd1b68288b3)

![Keypad to control remote servo](https://github.com/gloriahu28/Interactive-Lab-Hub/assets/142931503/3fa5109e-311a-42ad-b19e-8077e0e09ba7)

![Automatic sump pump control - pi in a basement w_ capacitance sensor sends message to pi w_ sump pump control servo attached to it when water level reaches certain height](https://github.com/gloriahu28/Interactive-Lab-Hub/assets/142931503/20210346-fcb1-4442-bb5e-24297c27a54f)

![Color Sensor](https://github.com/gloriahu28/Interactive-Lab-Hub/assets/142931503/9f90072e-421f-4589-b37d-db25767e78e1)

![Gossip Tracker](https://github.com/gloriahu28/Interactive-Lab-Hub/assets/142931503/660ab426-53cc-4355-be7c-a7a5afb83e21)

### Part C
### Streaming a Sensor

We have included an updated example from [lab 4](https://github.com/FAR-Lab/Interactive-Lab-Hub/tree/Fall2021/Lab%204) that streams the [capacitor sensor](https://learn.adafruit.com/adafruit-mpr121-gator) inputs over MQTT. 

Plug in the capacitive sensor board with the Qwiic connector. Use the alligator clips to connect a Twizzler (or any other things you used back in Lab 4) and run the example script:

<p float="left">
<img src="https://cdn-learn.adafruit.com/assets/assets/000/082/842/large1024/adafruit_products_4393_iso_ORIG_2019_10.jpg" height="150" />
<img src="https://cdn-shop.adafruit.com/970x728/4210-02.jpg" height="150">
<img src="https://cdn-learn.adafruit.com/guides/cropped_images/000/003/226/medium640/MPR121_top_angle.jpg?1609282424" height="150"/>
<img src="https://media.discordapp.net/attachments/679721816318803975/823299613812719666/PXL_20210321_205742253.jpg" height="150">
</p>

 ```
 (circuitpython) pi@raspberrypi:~ Interactive-Lab-Hub/Lab 6 $ python distributed_twizzlers_sender.py
 ...
 ```

**\*\*\*Include a picture of your setup here: what did you see on MQTT Explorer?\*\*\***
<img src='https://github.com/bensetel/Interactive-Lab-Hub/blob/Fall2023/Lab%206/imgs/IMG_2308.jpeg'>
<img src='https://github.com/bensetel/Interactive-Lab-Hub/blob/Fall2023/Lab%206/imgs/IMG_2309.jpeg'>

**\*\*\*Pick another part in your kit and try to implement the data streaming with it.\*\*\***

video:

https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/805b906e-7081-468e-9bd3-a027aa745b63




### Part D
### The One True ColorNet

It is with great fortitude and resilience that we shall worship at the altar of the *OneColor*. Through unity of the collective RGB, we too can find unity in our heart, minds and souls. With the help of machines, we can overthrow the bourgeoisie, get on the same wavelength (this was also a color pun) and establish [Fully Automated Luxury Communism](https://en.wikipedia.org/wiki/Fully_Automated_Luxury_Communism).

The first step on the path to *collective* enlightenment, plug the [APDS-9960 Proximity, Light, RGB, and Gesture Sensor](https://www.adafruit.com/product/3595) into the [MiniPiTFT Display](https://www.adafruit.com/product/4393). You are almost there!

<p float="left">
  <img src="https://cdn-learn.adafruit.com/assets/assets/000/082/842/large1024/adafruit_products_4393_iso_ORIG_2019_10.jpg" height="150" />
  <img src="https://cdn-shop.adafruit.com/970x728/4210-02.jpg" height="150">
  <img src="https://cdn-shop.adafruit.com/970x728/3595-03.jpg" height="150">
</p>


The second step to achieving our great enlightenment is to run `color.py`. We have talked about this sensor back in Lab 2 and Lab 4, this script is similar to what you have done before! Remember to activate the `circuitpython` virtual environment you have been using during this semester before running the script:

 ```
 (circuitpython) pi@raspberrypi:~ Interactive-Lab-Hub/Lab 6 $ systemctl stop mini-screen.service
 (circuitpython) pi@raspberrypi:~ Interactive-Lab-Hub/Lab 6 $ python color.py
 ...
 ```

By running the script, wou will find the two squares on the display. Half is showing an approximation of the output from the color sensor. The other half is up to the collective. Press the top button to share your color with the class. Your color is now our color, our color is now your color. We are one.

(A message from the previous TA, Ilan: I was not super careful with handling the loop so you may need to press more than once if the timing isn't quite right. Also, I haven't load-tested it so things might just immediately break when everyone pushes the button at once.)

**\*\*\*Can you set up the script that can read the color anyone else publish and display it on your screen?\*\*\***

Answer:
In order to get our colorreader.py script to correctly display the color, we had to modify color.py to include the alpha value in the message sent to the broker. After that, we created colorreader.py to read any messages sent to IDD/colors, parsed it into ints, and then converted it into RGB using the same math in color.py, and finally displaying this color to the screen.

https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/a710c57c-e0a1-451f-ba6a-06b7af899a0f

### Part E
### Make it your own

Find at least one class (more are okay) partner, and design a distributed application together based on the exercise we asked you to do in this lab.

**\*\*\*1. Explain your design\*\*\***     

**Purpose:**   
  The design is for a network of Raspberry Pis equipped with microphones, distributed across different locations. This system is intended to listen for specific keywords or phrases. It can serve various purposes, like monitoring for security keywords in a sensitive area, detecting specific topics in a research setting, or even for interactive art installations.

**Why It's Useful:**     
  This kind of system can be invaluable in scenarios where real-time monitoring of certain keywords is necessary. For instance, in a security context, it could listen for words like "help" or "emergency." In a business setting, it might monitor customer feedback or mentions of specific products.


**\*\*\*2. Diagram the architecture of the system.\*\*\***      


**Components:**  
- Raspberry Pi Units: These are the core of the system, each equipped with a microphone and placed in various locations.  
- Central Control System: This is where the data from all Raspberry Pis is collected and processed. It could be a server or another Raspberry Pi.  

**Process Flow:**  
  Input: Audio input from the environment is captured by microphones attached to each Raspberry Pi.  

**Computation:**  
- Local Processing: Each Raspberry Pi processes the audio to detect the specific keywords.  
- Central Processing: The central control system receives alerts from each Raspberry Pi when keywords are detected.  

**Output:**  
- Local Output: Each Raspberry Pi might give a local indication (like a light) when it detects a keyword.  
- Central Output: The central control system logs the alerts, potentially triggering a response or notification.  

**Connections:**  
- Raspberry Pis are connected to the central control system via a network (Wi-Fi or Ethernet).  
- The central control system may be connected to a broader network for remote monitoring or notifications.  



**\*\*\*3. Build a working prototype of the system.\*\*\***     

**Steps:**  
- Set up the Raspberry Pis with microphones and install speech recognition software.  
- Program each Pi to detect certain keywords and send an alert to the central system when these are detected.  
- Establish a network connection for communication between the Raspberry Pis and the central control system.  
- Develop a user interface for the central control system to monitor alerts and possibly interact with the system.  

**User Interface Considerations:**    
- The Raspberry Pis themselves don't need a direct user interface since they operate passively.    
- The central control system should have an intuitive interface for monitoring and managing alerts.    
- Indicators on the Raspberry Pis (like LEDs) could signal when they are active or have detected a keyword, providing some level of interaction for those in their vicinity.    

**Testing and Refinement:**
- Test the system in various environments to ensure the Raspberry Pis accurately detect the keywords and successfully communicate with the central system.  
- Adjust sensitivity and filtering to reduce false positives and negatives.  
- Ensure the system's user interface on the central control is clear and informative.  




**\*\*\*4. Document the working prototype in use.\*\*\*** It may be helpful to record a Zoom session where you should the input in one location clearly causing response in another location.


The three main scripts that are used in this demonstration are called "detect_word.py", "detector_script.py" and "lab6.sh". 

https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/22ede9b4-7280-4844-a58c-1b1457369398



<!--**\*\*\*5. BONUS (Wendy didn't approve this so you should probably ignore it)\*\*\*** get the whole class to run your code and make your distributed system BIGGER.-->
