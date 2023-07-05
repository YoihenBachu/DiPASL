# DiPASL: The Different Perspective ASL

This page is dedicated to tackle the problem that arises from developing a friendly device that can recognise
hand sign languages from the first-person perspective or in simpler terms, the ability to recognise the hand
signs from the back of the hand using CNN models

In this approach, we demonstrate the use of two different datasets (DiPASL-S900 and DiPASL-T900) to verify and
benchmark our findings. Anybody wishing to replicate this approach can follow the following instructions to 
get the datasets or collect your own datasets in the system that we propose to start from scratch.

## Setup

#### 1. Getting the codes
Go to the directory that you want to place the codes
This can be done by following the steps below
```
E:
cd <path to working directory>
```
This considers your working directory to be in E-drive, you can set it as you want.
Clone the repository in your local or cloud workspace (working directory) by using the command below
```
git clone https://github.com/YoihenBachu/DiPASL.git
```

#### 2. Creating conda environment
Create a new environment in your conda terminal
```
conda create --name DiPASL python=3.8
```
You can also use python 3.9 or 3.10

Go to the environment by using
```
conda activate DiPASL
```
You can also de-activate the environment by using 
```
conda deactivate DiPASL
```

Once inside the working environment, you can install the requirements given below. 
Note that installing requirements only needs you to be in the conda environment you want to work with
It doesn't need you to change your directory. But to run codes, you must change your working directory.
To change your working directory, follow step 1.

#### 3. Setting up requirements
You will find *requirements.txt* among the cloned files in the working directory.
If you are already in the working directory, ignore. Else, go to working directory. 
Install the pre-requisite libraries by using the command
```
pip install requirements.txt
```

#### 4. Library manipulations
Go to the ***HandTrackingModule.py*** in the cvzone package.
Under the ***HandDetector*** class, comment out or delete *line 91 to 94* in the ***findHands()*** method.\
**This is necessary to put skeleton maps in the detected hands and not bounding boxes**



