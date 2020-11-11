# NDCV-facial-keypoints-detection

In this project we will try to ombine our knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face!
We used Hacascade from OpenCV to detect face's location, then we cropped the face region and fed it to our CNN+FC network to perform keypoints regression. 

For this project, we'll using [YouTube Faces Datasets](https://www.cs.tau.ac.il/~wolf/ytfaces/) which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated keypoints.

Result:
<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-facial-keypoints-detection/blob/master/images/landmarks_numbered.jpg height="400" width="480">
</p>

## Demo
Face detection result using Haarcascade
<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-facial-keypoints-detection/blob/master/images/haar_cascade.png>
</p>

Facial keypoints detection result
<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-facial-keypoints-detection/blob/master/images/result1.png> <img src=https://github.com/Oktafsurya/NDCV-facial-keypoints-detection/blob/master/images/result2.png>
</p>

## Training vs validation Chart
We applied the early stopping method during the training and validation process. By using early stopping, we can get the best epoch that gives the best result from our model. Since Pytorch doesn't have a built-in function to perform early stopping so I made new class `EarlyStopping`

<p align="center"> 
<img src=https://github.com/Oktafsurya/NDCV-facial-keypoints-detection/blob/master/loss_plot.png>
</p>
