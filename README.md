# AWPCN
The python code for **Adaptive Weight Part-based Convolutional Network for Person Re-Identification**

**Abstract:** While part-based methods have been shown effective in person re-identification
task, it is unreasonable for most of them to treat each part equally, due to the retrieved
image maybe affected by deformation, occlusion and other factors, which makes the feature
information of some parts unreliable. Instead of using the same weight of each part
for the final person re-ID, we consider using an adaptive weight based on the part image
information for each part for precise person retrieve. Specifically, we aim at learning discriminative
part-informed features and propose an adaptive weight part-based convolutional
network (AWPCN) for person re-ID task. The core component of our AWPCN framework
is an adaptive weight model, in which the part-based convolutional network and the adaptive
weight model are used for feature refinement and feature-pair alignment, respectively.
Given an image input at first, it outputs a convolutional descriptor consisting of several
part-level features by the part-based convolutional network. And then, the corresponding
weights of each part are determined by the adaptive weight model. Finally, we can use the
adaptive weight part-based convolutional network joint to train each part loss and simultaneous
optimization of its feature representations. We evaluate the proposed AWPCN model
on Market-1501, DukeMTMC-reID and CUHK03 datasets. In extensive experiments, the
AWPCN model outperforms most of the state-of-the-art methods on these representative
datasets which clearly demonstrates the effectiveness of our proposed method.

The matlab code for AWPCN can be downloaded [here[google]]() or [here[baidu(password)]]().

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- [Optional] apex 

## Train
python train.py

## Test
python test.py

## Demo
python demo.py --query_index 666

## Results
| Dataste | Market-1501 | DukeMTMC-ReID | CUHK03 |
| --------| --------| ------- | ------ |
| Rank-1  | 0.940   | 0.857   | 0.673  | 
| mAP     | 0.821   | 0.741   | 0.628  |


## Contact
Feedbacks and comments are welcome! Feel free to contact us via dyuanhit@gmail.com


