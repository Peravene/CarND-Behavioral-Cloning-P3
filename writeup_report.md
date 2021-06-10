# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[imageArch]: ./doc/model.png "Model Visualization"
[videoLap]: ./video.mp4 "FULL Lap Auto-Steer"
[imageLoss]: ./doc/loss.png "Loss"
[imageCrop]: ./doc/cropping.png "Crop sky and engine hood"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* load.py for loading the images and labels into a pickle file
  * At the end generator was used instead in the model.py
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
*HINT: to enable Keras with GPU on my Laptop I had to install specific anaconda environment. A list of the package version can be found in: [./doc/conda_package_env.txt](conda_package_env.txt)*

#### 3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy
#### 1. Solution Design Approach

My strategy was to implement the content from the project lecture step by step to see hands on how iteratively the performance of the model improves. So first I verified the training and simulation pipeline by using only one flat and dense layer (model.py lines 109-110). This was only enough to keep the vehicle for a very short time in the lane but at least the framework was verified to be functional.

Then I implemented the already known LeNet architecture (model.py lines 115-122). With it the vehicle was at least driving in the straight road. But was not able to steer int the curve.

As in the course the NVIDIA self driving architecture was mentioned to be powerful, I have implemented the model from https://developer.nvidia.com/blog/deep-learning-self-driving-cars (model.py lines 127-137). Surprisingly it performed very well. With this architecture it was able to get the vehicle at least a full lap auto-steered.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20 ratio). Additionally I have plotted the mean squared error of the training and validation set: ![alt text][imageLoss]

At the end of the process, the vehicle is able to drive autonomously around boths track without leaving the road.

Here is a video of the recordinga:
- [video.mp4](./video.mp4)
- [video2.mp4](./video2.mp4)

We can still see that the MSE is lower than the one from the validation set. This implies that the model is overfitting. Fortunately not much that the first track is performing well. Anyway I have tested it on the second track where it was then failing.
#### 2. Final Model Architecture

Here is the summary of the used architecture.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              2459532
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0
_________________________________________________________________
```

And also a plot of it:
![alt text][imageArch]

#### 3. Model parameter tuning

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary (model.py line 145).
#### 4. Appropriate training data

I tried first with recording my own data set by driving one lap with the keyboard. But here i was failing to be able to keep the car at all in the lane with even the LeNet architecture. I assume that my control behavior with the keyboard is not well enough.

So to concentrate first on the pipeline I have used the provided dataset. In the generated [./data/IMG.mp4](./data/IMG.mp4) I could see that several laps in both directions have been driven.

The dataset consist of center, left and right images. I have used all of them. (load.py 27-61 or model.py33-56) For the left and right image an steering correction of 0.2 degree was sufficient. The center image was flipped to augment additionally.

After the collection process, I had 32144 number of data points. I then preprocessed this data by normalizing the RGB values. Also the area of sky and engine hood had been cropped out. (model.py 100-104) 
![alt text][imageCrop]

I finally randomly shuffled the data set and put 20% of the data into a validation set.


With the provided data set the vehicle was only autonomously steerable in the first track but not the second. So I knew that the model architecture itself was sufficient, but not the training data. Therefore I created my own dataset.

First I recorded on track one. I have driven it clockwise and anticlockwise several laps as much as possible in the lane center:
- ./ownDataFront/IMG/output.mp4
- ./ownDataBack/IMG/output.mp4

Additionally to train the model to return back from lane border to the lane center I have recorded those scenarios:
- ./ownDataReturn/IMG/output.mp4

As my goal was to increase also the performance on track two also there I recorded data for several laps.
- ./ownDataTrack2Front/IMG/output.mp4

Finally I had 20879 frames recorded which results in 83516 images by applying the data augmentation steps mentioned above.