# ----------------------------------------------------------------------
# Segmentation-Driven Quantization for Image Encoding
# 
# Project Title: Segmentation-Driven Quantization for Image Encoding
# # Project Description:
# Human visual systems are naturally drawn to areas of activity within
# images, typically containing objects such as people, animals, and vehicles.
# 
# This project implements a lossy encoder that uses image segmentation to
# guide the quantization process. Variable quantization, informed by
# segmentation, enhances compression efficiency while preserving high
# perceived quality for observers.
#
# Areas of low importance are quantized at a higher degree at the block level,
# achieving better compression without significantly affecting perceived quality.

import os
import csv
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Encode an image to a compressed binary file.")
# Define the input argument
# --input specifies the path to the input image file
parser.add_argument("--input", required=True, help="Path to the input image.")

# Define the output argument
# --output specifies the path to save the compressed binary file
parser.add_argument("--output", required=True, help="Path to the output binary file.")

# Define the adaptive encoding flag
# --adaptive is a boolean flag; if provided, it enables adaptive encoding
parser.add_argument("--adaptive", action="store_true", help="Enable adaptive encoding.")
args = parser.parse_args()

print(args.input)
# Define the JPEG quantization table
jpeg_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
"""
    Apply DCT (Discrete Cosine Transform) and quantization to an 8x8 block.
    
    Parameters:
    block (numpy.ndarray): The 8x8 block of image pixel values to process.
    quant_table (numpy.ndarray): The 8x8 quantization table to control compression.
    
    Returns:
    numpy.ndarray: The quantized block after applying DCT and dividing by the quantization table.
    """
    # Apply the Discrete Cosine Transform (DCT) to the block
    # Shift pixel values by subtracting 128 to center them around zero
def apply_dct_quantization(block, quant_table):
    """Apply DCT and quantization to an 8x8 block."""
    dct_block = cv2.dct(block.astype(np.float32) - 128)
    # Quantize the DCT coefficients by dividing them by the quantization table
    
    # Round the values to the nearest integer and cast to int32 for efficiency
    quantized_block = np.round(dct_block / quant_table).astype(np.int32)
    return quantized_block

"""
    Apply inverse DCT (Discrete Cosine Transform) and de-quantization to an 8x8 block.
    
    Parameters:
    block (numpy.ndarray): The quantized 8x8 block of DCT coefficients.
    quant_table (numpy.ndarray): The 8x8 quantization table used during encoding.
    
    Returns:
    numpy.ndarray: The reconstructed 8x8 block of pixel values, clipped to the valid range [0, 255].
    """
def inverse_dct_quantization(block, quant_table):
    """ Apply inverse DCT and de-quantization to an 8x8 block.
        De-quantize the block by multiplying with the quantization table
        This step reverses the effect of the quantization applied during encoding. """
    dequantized_block = block * quant_table
    """ Apply the inverse DCT (IDCT) to reconstruct the pixel values
        Add 128 to shift the pixel values back to the original range. """ 
    idct_block = cv2.idct(dequantized_block.astype(np.float32)) + 128
    return np.clip(idct_block, 0, 255).astype(np.uint8)

"""
    Calculate the bitrate based on the number of non-zero coefficients.
    
    Parameters:
    quantized_blocks (list of numpy.ndarray): A list of 8x8 quantized blocks.
    
    Returns:
    float: The bitrate, defined as the ratio of non-zero coefficients to total coefficients.
    """
def calculate_bitrate(quantized_blocks):
    """Calculate bitrate based on the number of non-zero coefficients."""
    total_coefficients = sum(block.size for block in quantized_blocks)
    non_zero_coefficients = sum(np.count_nonzero(block) for block in quantized_blocks)
    """ Clip the pixel values to the valid range [0, 255]
        Convert the result to uint8 for compatibility with image data formats. """
    return non_zero_coefficients / total_coefficients

# Input image path
image_path = str(args.input)
image = cv2.imread(image_path)

# Create output folder
output_folder = str(args.output)
os.makedirs(output_folder, exist_ok=True)

# Convert to YCrCb and compute edges
image_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
h, w, _ = image_ycc.shape
edges = cv2.Canny(image_ycc[:, :, 0], 100, 200)

# Identify blocks with edges
blocks_with_edges = np.zeros((h // 8, w // 8), dtype=bool)
for i in range(0, h, 8):
    for j in range(0, w, 8):
        block = edges[i:i+8, j:j+8]
        if np.any(block):  # Mark block if edges are present
            blocks_with_edges[i//8, j//8] = True

# Iterative encoding process
important_scale = 1.0
non_important_scale = 2.0
if bool(args.adaptive) == 1:
    pass_encoding = 10
else:
    pass_encoding = 1
    
# Define the threshold tolerance for bitrate similarity
threshold_tolerance = 0.001

# Initialize a list to store data for CSV export, including headers
csv_data = [["Pass", "Regular Bitrate", "Adaptive Bitrate", "Important Scale", "Non-Important Scale"]]

# Loop over the number of encoding passes
for _ in range(0, pass_encoding):
    # Lists to store reconstructed image data for each channel
    channels_regular = []
    channels_adaptive = []
    
    # Iterate over each color channel (Y, Cb, Cr)
    for c in range(3):
        channel = image_ycc[:, :, c] #### Extract the channel data from the YCC image
        #### Initialize arrays for reconstructed images
        reconstructed_regular = np.zeros_like(channel, dtype=np.uint8)
        reconstructed_adaptive = np.zeros_like(channel, dtype=np.uint8)
        #### Lists to store quantized blocks for regular and adaptive encoding
        regular_encoded_blocks = []
        adaptive_encoded_blocks = []

        # Iterate over 8x8 blocks in the image
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8]
                if block.shape != (8, 8):  # Skip incomplete blocks
                    continue

                '''# Regular encoding
                # Apply DCT and quantization using the standard JPEG quantization table'''
                quantized_block = apply_dct_quantization(block, jpeg_quantization_table)
                regular_encoded_blocks.append(quantized_block)
                '''# Reconstruct the block using inverse DCT and de-quantization'''
                reconstructed_regular[i:i+8, j:j+8] = inverse_dct_quantization(quantized_block, jpeg_quantization_table)

                # Adaptive encoding
                # Adjust the quantization table based on the presence of edges in the block
                if blocks_with_edges[i//8, j//8]:
                    adaptive_quant_table = jpeg_quantization_table * important_scale #### Check if the block contains edges
                else:
                    adaptive_quant_table = jpeg_quantization_table * non_important_scale
                
                # Apply DCT and quantization using the adaptive quantization table
                quantized_block_adaptive = apply_dct_quantization(block, adaptive_quant_table)
                adaptive_encoded_blocks.append(quantized_block_adaptive)
                # Reconstruct the block using inverse DCT and de-quantization
                reconstructed_adaptive[i:i+8, j:j+8] = inverse_dct_quantization(quantized_block_adaptive, adaptive_quant_table)
       
        # Append the reconstructed data for the current channel
        channels_regular.append(reconstructed_regular)
        channels_adaptive.append(reconstructed_adaptive)

    # Merge channels
    regular_encoded_image = cv2.merge(channels_regular)
    adaptive_encoded_image = cv2.merge(channels_adaptive)

    # Calculate bitrates
    regular_bitrate = calculate_bitrate(regular_encoded_blocks)
    adaptive_bitrate = calculate_bitrate(adaptive_encoded_blocks)
    

    # Adjust scales and calculate the bitrate difference
    bitrate_difference = abs(regular_bitrate - adaptive_bitrate)

    # Append the current iteration's data to the CSV data list
    # This includes iteration number, regular bitrate, adaptive bitrate, and scale factors
    csv_data.append([_ + 1, regular_bitrate, adaptive_bitrate, important_scale, non_important_scale])
    
    # Print the current pass details for debugging and tracking progress
    print(f"Pass {_ + 1}: Regular Bitrate = {regular_bitrate:.2f}, Adaptive Bitrate = {adaptive_bitrate:.2f}, Important Scale = {important_scale:.2f}, Non-Important Scale = {non_important_scale:.2f}")
            
    # Check if the difference between bitrates is within the defined tolerance threshold
    if bitrate_difference <= threshold_tolerance:
        print("Threshold reached -> Bitrates Similar.")
        break # Exit the loop since the desired bitrate similarity has been achieved
        
    # If the difference exceeds the tolerance threshold, adjust the scale factors
    if bitrate_difference > threshold_tolerance:
        if regular_bitrate > adaptive_bitrate:
            important_scale -= 0.05 #### Decrease the importance scale to bring adaptive bitrate closer to regular bitrate

        else:
            important_scale += 0.05 #### Increase the importance scale to bring adaptive bitrate closer to regular bitrate
        
    
    
    
cv2.imwrite(os.path.join(output_folder, f"regular_encoded_image_pass.jpg"), cv2.cvtColor(regular_encoded_image, cv2.COLOR_YCrCb2BGR))
cv2.imwrite(os.path.join(output_folder, f"adaptive_encoded_image_pass.jpg"), cv2.cvtColor(adaptive_encoded_image, cv2.COLOR_YCrCb2BGR))
cv2.imwrite(os.path.join(output_folder, "edges_signal.jpg"), (blocks_with_edges.astype(np.uint8) * 255))

# Save CSV report
csv_path = os.path.join(output_folder, "bitrate_report.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Images and report saved in folder: {output_folder}")

