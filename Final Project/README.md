## Project Name: Byte the Robot

**COLLABORATORS** 
Gilberto Ruiz, Gloria Hu, Ben Setel, Kenneth Lee, Yifan Yu, Michael Hanlon 

## Project Idea

For our final project, we built a small robot with the 3D printers here at the Makerspace and its purpose is to mimic a person using the Mediapipe library. One Pi device will run Mediapipe with a webcam to track the user’s movements, and digitally maps out their arms and head.  It will communicate this information with the Pi that controls the robot through MQTT and allows the robot to mimic the user's head, shoulders, elbows and hand movements through its servo motors.

## Description

The main idea behind our project is to create an “avatar” robot which mimics the movements of the user that is being tracked. We track the user’s movement through the use of a webcam and mediapipe, which digitally maps out the arms and head and provides this data in a form that we can use to control the robot in similar ways. This data about the user’s movement is translated into movements that the robot can make with its arms, which are enabled through its built-in servos and motors.

## Project Plan Highlights/Timeline
Nov 14: Project Plan Pitch

Nov 21: Design avator system and diagrams

Nov 28 & 30: Start 3D printing process

Dec 5 & 7: Assembling and iterating on the device

Dec 10: Continously working and making poster for presentation

Dec 12: Presenting in class

Dec 14: Wraping up and finishing the Report


### Documentation of Design Process

### Storyboard

![1](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/1b8ac4a3-84a9-42fd-8344-4d1a1c83fd96)  
![2](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/18042932-17ea-46dc-9e9c-cff6868527eb)  
![3](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/f60afbd8-3371-4508-b0b2-bb8f2179cc04) 

### Poster Process
![b6f2106e6f7d5c91f155fdd76238b3c](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/a4446f98-367d-4cb7-8c8f-ce5addfbfae4)  
![d2ffba73b0be27c9c078dc208cbd60b](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/382f2229-e6dd-430f-a52b-fe33231c9cb8)  
![55c7f348d5bed20074800b228a70e3b](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/759700f1-8d93-4b29-8f45-11b15fd19634)  
![ada757237bc4046b347b3a7564d28f7](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/72a92430-30c7-40e4-b5b2-89f69fc2fbb3)  


## Robot Design and Parts Needed:

### Source
The stl files to 3D print the robot was found online in the following link: 
[Robot](https://www.kevsrobots.com/blog/chip.html#download-the-3d-printable-stl-files)

### How Byte was 3D Printed
Byte the Robot had his parts manufactured in the Makerspace using the UltiMaker 3D printer. The blueprint for Byte came from a 3D model designed by Kevin McAleer, through the use of his publicly available STL file. The blueprint was perfect because it already accommodates space for servos in the arms and head, and also includes holes where screws were then used to connect all of Byte’s body parts together.

With the stl files, we were able to print all the pieces to build the robot and these were the results:  

<img width="514" alt="d42f31d152769bb6e08ec8afa3e4684" src="https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/1c4c49c0-52f6-465a-b751-67eb9f8cefcb">

We made the decision to not print the legs of the robot and focus more on the head and arm movement due to time constraints and resources with servos.    
  

### Servo Motor Wiring:
After building the robot with all its servos, we now needed to wire the servos to the following servo motor driver:  

<img width="306" alt="4d3e5ea447525edca4f25e946e3d7b5" src="https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/fa2e7883-995e-4e2b-afb3-d211298f077a">

And with this driver, this is where we assigned each servo motor:  

<img width="760" alt="0001496166b1d40e21a325a81380f4b" src="https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/d6e7dca0-2eb6-4e93-8d57-196ef0bc20ff">


| Servo Driver Port | Motor Assignment |
|-----------------|-----------------|
| 0 | Left Hand |
| 1 | Left Elbow |
| 2 | Left Shoulder |
| 3 | Head |
| 12 | Right Shoulder |
| 13 | Right Elbow |
| 14 | Right Hand |

## Design Process Screenshots
![eddd54170e998ff222fc104335fff1d](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/31565358-5a18-475d-9d92-361e5b3b5a2b)
![54d8c7296dae7ee7a5d1ea01736d666](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/10268f97-46fe-442f-8792-ce6dce033841)
![5b03da29f1da1d3da1657818feadfb2](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/11d415c2-ba11-4585-bd0f-40e10589fe4a)
![f9395075577efdc3fbec3b970d13043](https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/126a76ae-e42c-433f-a676-23e5f0c24acf)





## Archive of all code
Archive of all code can be found here:

https://github.com/Ruiznogueras05CT/Interactive-Lab-Hub/tree/Fall2023/Final%20Project



## Video and Testing  



https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/5c53b079-8878-464b-810d-e86cef55cda4



https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/80d2ccee-e5e0-4657-b198-06718288c12d



https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/d3c26350-4bf2-45df-82a3-7092e04e4214




https://github.com/yifanwow/Interactive-Lab-Hub/assets/64716158/e1e67069-e748-499f-8d04-d40c448f5bd3



## Reflections on Process (What have we learned or wish we knew at the start?)
Reflecting on our robot design project at the Makerspace, we created a small robot capable of mimicking human movements. Utilizing 3D printing technology, our primary objective was to construct an “avatar” robot. This robot was designed to replicate a person's movements, tracked and analyzed through a webcam using the Mediapipe library. The setup involved two key components: one Raspberry Pi device ran Mediapipe to capture and process human movements, and another controlled the robot, allowing it to mimic the user's head, shoulders, elbows, and hand movements with servo motors.

Our project centered around the idea of creating a tangible, responsive entity that could interact with humans in a unique and innovative way. We aimed to bridge the gap between digital motion capture and physical replication, translating digital data into physical actions. The main challenge was ensuring that the robot could accurately and fluidly mimic human movements, a task that involved complex programming and mechanical design.

Reflecting on our achievements and the challenges we faced, we identified several areas for improvement in future iterations of the project. Firstly, we recognized the need for a more advanced robot design, particularly with articulated elbows. Our initial model had limited joint mobility, which restricted the robot's ability to perform complex human movements. By upgrading to a newer version of the robot with more sophisticated joint mechanisms, we could enhance the robot's capacity for more accurate and life-like motion replication.

Another area for improvement was the servo movement. The servos we used initially resulted in somewhat jerky and mechanical movements. For a more realistic and human-like motion, smoother servo movements are essential. This could be achieved through better quality servos or by refining our control algorithms to allow for more fluid transitions and movements.
In terms of hand tracking, while we successfully implemented basic Mediapipe tracking, there's room for incorporating more advanced hand tracking features. This improvement would enable finer control over the robot's hand movements, allowing for more detailed and precise mimicry of the user's gestures.

Lastly, exploring alternative speech-to-text models could significantly enhance the project. We used the Vosk model for our initial prototype, but other models might offer improved accuracy or functionality. This change could lead to better responsiveness and interaction capabilities of the robot, making it more effective in understanding and executing voice commands.

In conclusion, our project was a great learning experience that pushed the boundaries of our understanding in robotics and human-computer interaction. During user testing and presentations, we discovered that Byte could serve as a social bot because the small details of its movements resonated with real-life human interactions. We found this project meaningful as it inspired us to interact, feel, and understand others more deeply, building more connections and enhancing our empathy.

## Future Improvements
### Integration of a More Advanced Robot Design:
Current State: Our initial model had limited joint mobility, particularly in the elbows.

Improvement: In future iterations, we propose using a newer version of the robot with articulated elbows. This would enhance the robot's ability to mimic complex human movements more accurately.

### Enhancing Servo Movement:
Current State: The servos used had a somewhat jerky movement, affecting the fluidity of the robot's motions.

Improvement: Implementing smoother servo movements could make the robot's actions more lifelike and less mechanical.

### Advanced Hand Tracking:
Current State: Our current model used basic Mediapipe tracking for hand movements.

Improvement: Incorporating advanced hand tracking features, possibly by integrating additional Mediapipe functionalities, would allow for more precise control of the robot's hands.

### Exploring Alternative Speech-to-Text Models:

Current State: We utilized the Vosk model for speech-to-text conversion.

Improvement: Investigating other speech-to-text models could offer improved accuracy or efficiency, enhancing the robot's responsiveness to vocal commands.

## Groupwork Distribution
Ben, Gil, Michael: Programming, assembling and 3D printing Robot

Gloria: Designing and 3D printing Robot

Kenneth: 3D Printing and testing

Yifan: 3D Printing and documenting
