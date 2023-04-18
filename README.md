# TrafficPredict

Code for TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents (AAAI), Oral, 2019

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
