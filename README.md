# Segmentation-Driven-Quantization-for-Image-Encoding
The human visual system focuses on active areas in images, such as people or vehicles. This project introduces a lossy encoder using image segmentation to guide quantization, improving compression while preserving quality. Less critical areas undergo higher quantization at the block level. The diagram below illustrates the process.

## Team Members
- **Juan Merlos**  
- **Hadise Pishdast**

---

## Overview
This project implements a lossy image encoding process that leverages **image segmentation** to guide quantization. The key idea is to apply adaptive quantization based on the importance of different regions within the image. By focusing on preserving high-quality details in areas with significant visual activity (such as objects or edges) and allowing more aggressive compression in less important areas, this approach improves compression efficiency while maintaining overall image quality.

The program outputs two versions of the encoded image:
1. **Regular Encoding**: Standard JPEG quantization applied uniformly.
2. **Adaptive Encoding**: Segmentation-based quantization with variable quality across regions.

---

## Features
- Edge detection to identify important areas of the image.
- Adaptive quantization scales for important and non-important blocks.
- Adjustable compression settings to balance bitrate and quality.
- Visualization of original, segmented, and encoded images.
- Bitrate comparison between regular and adaptive encoding.

---

## Software Requirements
- **Python 3.8+**
- Libraries:
  - `cv2` (OpenCV)
  - `numpy`
  - `matplotlib`

---

## How to Install and Run
### 1. Installation
- Clone this repository:  
  ```bash
  git clone <repository-link>
  cd <repository-folder>
