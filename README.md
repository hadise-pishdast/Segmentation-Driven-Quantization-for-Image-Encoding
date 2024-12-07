# Segmentation-Driven Quantization for Image Encoding

## Project Title
Segmentation-Driven Quantization for Image Encoding

## Team Members
- Juan Merlos
- Hadise Pishdast

## Project Description
Human visual systems are naturally drawn to areas of activity within images, typically containing objects such as people, animals, and vehicles. This project proposes the implementation of a lossy encoder that leverages image segmentation to guide the quantization process. By applying variable quantization informed by segmentation, the goal is to enhance compression efficiency while preserving high image quality for the observer.

Areas deemed of low importance are quantized to a higher degree at the block level, thereby achieving better compression without significantly compromising perceived quality.

## Features
- Implementation of image segmentation for adaptive quantization.
- Variable quantization scales for important and non-important image blocks.
- Iterative adjustment of quantization to minimize bitrate differences.
- Visualization of original, regular encoding, adaptive encoding, and edge detection results.

## Software Used
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Installation Guide

### Prerequisites
1. Python 3.8 or later
2. A Jupyter Notebook environment
3. Libraries: OpenCV, NumPy, Matplotlib

Install the required libraries using:
```bash
pip install opencv-python numpy matplotlib
```

### Steps to Run
1. Clone this repository to your local machine.
2. Open the Jupyter Notebook in your preferred environment.
3. Provide an input image in the specified directory (refer to the **Input Format** section).
4. Execute the notebook cell by cell to process the image.
5. The outputs will be saved as `regular_encoded_image.jpg` and `adaptive_encoded_image.jpg` in the current directory. Plots of results will also be displayed inline.

## Input Format
- **Format:** `.png` images
- **Resolution:** The input image must be 1920x1080 pixels.
- **Color Space:** Images should be in RGB.

## Operating System Compatibility
This project has been tested on:
- Linux
- Windows

## Expected Output
1. `regular_encoded_image.jpg`: Image compressed with regular encoding.
2. `adaptive_encoded_image.jpg`: Image compressed with segmentation-driven adaptive encoding.
3. Visualization of:
   - Original image
   - Edges/Contours detected in the image
   - Reconstructed image from regular encoding
   - Reconstructed image from adaptive encoding
   - Blocks with edges highlighted

## Code Workflow
1. **Segmentation**: Edge detection is performed using the Canny algorithm to identify areas of importance.
2. **Quantization**:
   - Regular encoding uses a standard JPEG quantization table.
   - Adaptive encoding uses different quantization scales for blocks identified as important or non-important.
3. **Bitrate Calculation**: Bitrate is calculated based on the number of non-zero coefficients in quantized blocks.
4. **Iterative Adjustment**: Quantization scales are adjusted iteratively to minimize the bitrate difference between regular and adaptive encoding.
5. **Visualization**: The results, including images and bitrates, are displayed and saved for analysis.

## References
1. OpenCV Documentation: [https://opencv.org/](https://opencv.org/)
2. NumPy Documentation: [https://numpy.org/](https://numpy.org/)
3. JPEG Compression Basics: [https://en.wikipedia.org/wiki/JPEG](https://en.wikipedia.org/wiki/JPEG)

## Code Comments
The code is well-commented to provide clarity on each function and its purpose. Key areas include:
- DCT and quantization functions
- Edge detection and block segmentation logic
- Iterative adjustment of quantization scales
