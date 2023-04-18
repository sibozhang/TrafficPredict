# TrafficPredict

Code for TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents (AAAI), Oral, 2019. [Project Page](http://gamma.cs.unc.edu/TPredict/TrafficPredict.html) 

## Introduction

To safely and efficiently navigate in complex urban traffic, autonomous vehicles must make responsible predictions in relation to surrounding traffic-agents (vehicles, bicycles, pedestrians, etc.). A challenging and critical task is to explore the movement patterns of different traffic-agents and predict their future trajectories accurately to help the autonomous vehicle make reasonable navigation decision. To solve this problem, we propose a long short-term memory-based (LSTM-based) realtime traffic prediction algorithm, TrafficPredict. Our approach uses an instance layer to learn instances' movements and interactions and has a category layer to learn the similarities of instances belonging to the same type to refine the prediction. In order to evaluate its performance, we collected trajectory datasets in a large city consisting of varying conditions and traffic densities. The dataset includes many challenging scenarios where vehicles, bicycles, and pedestrians move among one another. We evaluate the performance of TrafficPredict on our new dataset and highlight its higher accuracy for trajectory prediction by comparing with prior prediction methods.

## Data preparation

The trajectory dataset consists of 53min training sequences and 50min testing sequences captured at 2 frames per second.

object counts for cars, bicycles, and pedestrians are as follows (https://arxiv.org/pdf/1811.02146.pdf): 
16.2k, 5.5k, 60.1k

Sample Data:
[sample_trajectory.zip](https://ad-apolloscape.cdn.bcebos.com/trajectory/sample_trajectory.zip)
[sample_image.zip](https://ad-apolloscape.cdn.bcebos.com/trajectory/sample_image.zip)

Full data:
[prediction_train.zip](https://ad-apolloscape.cdn.bcebos.com/trajectory/prediction_train.zip)
[prediction_test.zip](https://ad-apolloscape.cdn.bcebos.com/trajectory/prediction_test.zip)
or
```
wget https://ad-apolloscape.cdn.bcebos.com/trajectory/prediction_train.zip
wget https://ad-apolloscape.cdn.bcebos.com/trajectory/prediction_test.zip

```

## Data Structure

The folder structure of the trajectory prediction is as follows:

1. prediction_train.zip: training data for trajectory prediction.
   * Each file is a 1min sequence with 2fps.
   * Each line in a file contains frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading.
   * There are five different object types as shown in following table. During the evaluation in this challenge, we treat the first two types, small vehicle and big vehicle, as one type (vehicle).
   
| object_type 	| small vehicles 	| big vehicles 	| pedestrian 	| motorcyclist and bicyclist 	| others 	|
|-------------	|----------------	|--------------	|------------	|----------------------------	|--------	|
| ID          	| 1              	| 2            	| 3          	| 4                          	| 5      	|

   * Position is given in the world coordinate system. The unit for the position and bounding box is meter.
   * The heading value is the steering radian with respect to the direction of the object.
   * In this challenge, we mainly evaluate predicted position_x and position_y in the next 3 seconds.
   
2. prediction_test.zip: testing data for trajectory prediction.
   * Each line contains frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading.

   * A testing sequence contains every six frames in the prediction_test.txt. Each sequence is evaluated independently.
   
# How to Run
```
cd srnn
```
Train:
```
python train.py 
```
Test:
```
python sample.py --epoch=n 
```
where n is the epoch at which you want to load the saved model. (See the code to understand all the arguments that can be given to the command)

## Baseline result:

| Rank     | Method         | WSADE  | ADEv   | ADEp   | ADEb    | WSFDE   | FDEv    | FDEp    | FDEb    |
|----------|----------------|--------|--------|--------|---------|---------|---------|---------|---------|
| Baseline | Trafficpredict | 8.5881 | 7.9467 | 7.1811 | 12.8805 | 24.2262 | 12.7757 | 11.1210 | 22.7912 |

## Publication
[![Depth Guided Video Inpainting for Autonomous Driving](https://res.cloudinary.com/marcomontalbano/image/upload/v1595308447/video_to_markdown/images/youtube--dST6NDxEMU8-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=dST6NDxEMU8 "Depth Guided Video Inpainting for Autonomous Driving")

Please cite our paper in your publications if our dataset is used in your research.

TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents. [PDF](https://arxiv.org/abs/1811.02146)
[BibTex](https://ad-apolloscape.cdn.bcebos.com/TrafficPredict/trafficpredict_bibtex.txt) [Website](http://gamma.cs.unc.edu/TPredict/TrafficPredict.html)

Yuexin Ma, Xinge Zhu, Sibo Zhang, Ruigang Yang, Wenping Wang, and Dinesh Manocha.

AAAI(oral), 2019

```
@inproceedings{ma2019trafficpredict,
  title={Trafficpredict: Trajectory prediction for heterogeneous traffic-agents},
  author={Ma, Yuexin and Zhu, Xinge and Zhang, Sibo and Yang, Ruigang and Wang, Wenping and Manocha, Dinesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={6120--6127},
  year={2019}
}
```
