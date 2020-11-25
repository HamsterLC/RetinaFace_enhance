This repo is an improvement version of https://github.com/biubug6/Pytorch_Retinaface

Improvements in is repo:
1. More Data Augmentation(Expand, MinIOU Crop with different ratio, Multi-Scale Training)
2. Long Training Schedule with more epoch and stage
3. Support More Backbone(MV1_0.25, MV2_0.35, MV2_1.0)

# Comparison with origin implement(original image scale)
| Style | BackBone | MAdds | Parameters | easy | medium | hard |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Enhance_MV1 | MV1_0.25 | 1.0B | 0.44M | 91.8% | 90.2% | 76.4% |
| Enhance_MV2 | MV2_0.35 | 1.0B | 0.48M | 92.8% | 90.9% | 79.4% |
| RetinaFace_Ori[6] | MV1_0.25 | 1.0B | 0.44M | 89.6% | 87.1% | 69.1% |
| RetinaFace_biubug6[7] | MV1_0.25 | 1.0B | 0.44M | 90.7% | 88.2% | 73.8% |

AP on hard set can surpass 80% by using knowledege distilling.
Some deploy models are in the Deploy folder.


# Reference
1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) 
2. [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
3. [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
4. [ASFD: Automatic and Scalable Face Detector](https://arxiv.org/abs/2003.11228) 
5. [CenterFace](https://github.com/Star-Clouds/CenterFace)
6. [RetinaFace(insightface)](https://github.com/deepinsight/insightface)
7. [RetinaFace(biubug6)](https://github.com/biubug6/Pytorch_Retinaface)
8. [RetinaFace(lbin)](https://github.com/lbin/Retinaface_Mobilenet_Pytorch)

