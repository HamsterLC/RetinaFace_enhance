# Face-Detection-Lite(Under Construction)
Here are some lightweight face detection model designed for a project in SmartSens.
We take MobileNetV2[1] as backbone.

# About SmartSens
Smartsens technology is a high-performance CMOS image sensor chip design company founded in 2011 with a global leading R & D team in Shanghai, Beijing and other regions of China.

# Comparison with other lightweight face detection model
| Style | MAdds | Parameters | easy | medium | hard |
|:-|:-:|:-:|:-:|:-:|:-:|
| Smart_V0 | 0.27B | 0.10M | 84.9% | 81.4% | 64.0% |
| |
| Smart_V1* | 0.67B | 0.27M | 89.8% | 84.9% | 53.8% |
| Ultra_1M_RBF*[2] |   | ~0.3M | 85.5% | 82.2% | 57.9% |
| libfacedetection_v2*[3] |  | 0.8M | 77.3% | 71.8% | 48.5% |
| |
| Smart_V1 | 0.67B | 0.27M | 91.0% | 88.8% | 75.4% |
| ASFD_D0[4] | 0.73B | 0.62M | 90.1% | 87.5% | 74.4% |
| |
| Smart_V2 | 1.1B | 0.48M | 92.8% | 90.9% | 79.4% |
| CenterFace[5] |  | 1.8M | 92.2% | 91.1% | 78.2% |
| RetinaFace_Ori[6] | 1.0B | 0.44M | 89.6% | 87.1% | 69.1% |
| RetinaFace_biubug6[7] | 1.0B | 0.44M | 90.7% | 88.2% | 73.8% |
| |
| Smart_V3 | 1.17B | 0.59M | 93.3% | 91.5% | 80.5% |
| RetinaFace_libin(DCN)[8] |  | 0.65M | 92.7% | 90.9% | 80.2% |
| |
| Smart_V4 | 3.89G | 2.26M | 93.9% | 92.9% | 83.7% |
| ASFD_D1[4] | 4.27G | 3.9M | 93.3% | 91.7% | 83.6% |
* Use 640x640 as input   The other use origin

# Reference
1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) 
2. [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
3. [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
4. [ASFD: Automatic and Scalable Face Detector](https://arxiv.org/abs/2003.11228) 
5. [CenterFace](https://github.com/Star-Clouds/CenterFace)
6. [RetinaFace(insightface)](https://github.com/deepinsight/insightface)
7. [RetinaFace(biubug6)](https://github.com/biubug6/Pytorch_Retinaface)
8. [RetinaFace(lbin)](https://github.com/lbin/Retinaface_Mobilenet_Pytorch)

