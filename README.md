# DiPASL the Different Perspective ASL

This page is dedicated to tackle the problem that arises from developing a friendly device that can recognise
hand sign languages from the first-person perspective or in simpler terms, the ability to recognise the hand
signs from the back of the hand using CNN models

In this approach, we demonstrate the use of two different datasets (DiPASL-S900 and DiPASL-T900) to verify and
benchmark our findings. Anybody wishing to replicate this approach can follow the following instructions to 
get the datasets or collect your own datasets in the system that we propose to start from scratch.

## Setup

1. First clone the repository in your local or cloud workspace by using the command below

```
git clone https://github.com/YoihenBachu/DiPASL.git
```

1. Create a new environment in your conda terminal

```
conda create --name DiPASL python=3.8
```
You can also use python 3.9 or 3.10

1. Install the pre-requisite libraries by using the command

```
pip install requirement.txt
```

1. Remove this two lines from the cvzone.findhands() module