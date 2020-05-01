# Face-Detection-Lite
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
| Ultra_1M_RBF* |   | ~0.3M | 85.5% | 82.2% | 57.9% |
| libfacedetection_v2* |  | 0.8M | 77.3% | 71.8% | 48.5% |
| |
| Smart_V1 | 0.67B | 0.27M | 91.0% | 88.8% | 75.4% |
| ASFD_D0 | 0.73B | 0.62M | 90.1% | 87.5% | 74.4% |
| |
| Smart_V2 | 1.1B | 0.48M | 92.8% | 90.9% | 79.4% |
| CenterFace |  | 1.8M | 92.2% | 91.1% | 78.2% |
| RetinaFace_Ori | 1.0B | 0.44M | 89.6% | 87.1% | 69.1% |
| RetinaFace_biubug | 1.0B | 0.44M | 90.7% | 88.2% | 73.8% |
| |
| Smart_V3 | 1.17B | 0.59M | 93.3% | 91.5% | 80.5% |
| RetinaFace_libin(DCN) |  | 0.65M | 92.7% | 90.9% | 80.2% |
| |
| Smart_V4 | 3.89G | 2.26M | 93.9% | 92.9% | 83.7% |
| ASFD_D1 | 4.27G | 3.9M | 93.3% | 91.7% | 83.6% |
* Use 640x640 as input   The other use origin

#Reference
