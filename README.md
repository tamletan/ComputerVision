# Project: Build an application for Attendance Tracking System
Author: Le Tan Tam,
        Dang Phuc Thinh,
        Nguyen Huu Phat
        
March-June 2020
Major: Information Technology - Computer Vision
University: Danang University of Sciences and Technology

# How to install
- Dependencies : You can use Linux, Windows or MacOS. MacOS for best experience. But in this project, we use Windows to do this project.
- Clone this directory by <code> git clone</code>or Download the zip.

# How to use
- After fishing clone or download the zip, you should extract file(if you <code>Download the zip</code>).

1. Create model to train Liveness model. This model will be use for <code>anti-spoofing face</code>.
Open folder <code>liveness</code>, you put your videos which recorded your faces to create data in <code>videos</code> folder. Open terminal and try this: 

<b>Step 1</b>
- python gather.py -t real
- python gather.py -t fake<br>
These command will excute file <code>gather.py</code>. After excuting, we will have <code>dataset</code> about liveness.

<b>Step 2</b>
- python train.py<br>
This command will excute file <code>train.py</code>. After excuting, we will have <code>model</code> about liveness.

<b>Step 3</b>
- python demo.py<br> 
This command will excute file <code>train.py</code>. After excuting, we will have <code>model</code> about liveness. You can check your data is correct or not.

2. After create <code>liveness</code> model. You should back to previous folder and try this:<br>
First of all, you should prepare dataset for each person who want to regconition. You need to prepare 15-30 pictures for each person. Create <code>students.csv</code> contains students list.
After preparing, try this to run program below:

<b>Step 1</b>
- python face_detection_train.py<br>
This command will excute file <code>face_detection_train.py</code>. After excuting, we will have <code>models.h5</code> about face regconition.

<b>Step 2</b>
- python face_detection_testimage.py or face_detection_testvideo.py <br>
This command is used to verify the accuracy of models. 

<b>Step 3</b>
- python demo.py <br>
This command is used to <code>run</code> the program.

# Troubleshoot
If there is any trouble, please feel free to contact with me via email: dangphucthinha9@gmail.com or tamletannk94@gmail.com

# Hope you like this !!!

