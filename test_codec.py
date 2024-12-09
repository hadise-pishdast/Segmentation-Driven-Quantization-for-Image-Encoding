import os
import csv
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Encode an image to a compressed binary file.")
parser.add_argument("--input", required=True, help="Path to the input image.")
parser.add_argument("--output", required=True, help="Path to the output binary file.")
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

def apply_dct_quantization(block, quant_table):
    """Apply DCT and quantization to an 8x8 block."""
    dct_block = cv2.dct(block.astype(np.float32) - 128)
    quantized_block = np.round(dct_block / quant_table).astype(np.int32)
    return quantized_block

def inverse_dct_quantization(block, quant_table):
    """Apply inverse DCT and de-quantization to an 8x8 block."""
    dequantized_block = block * quant_table
    idct_block = cv2.idct(dequantized_block.astype(np.float32)) + 128
    return np.clip(idct_block, 0, 255).astype(np.uint8)

def calculate_bitrate(quantized_blocks):
    """Calculate bitrate based on the number of non-zero coefficients."""
    total_coefficients = sum(block.size for block in quantized_blocks)
    non_zero_coefficients = sum(np.count_nonzero(block) for block in quantized_blocks)
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
    
threshold_tolerance = 0.001

csv_data = [["Pass", "Regular Bitrate", "Adaptive Bitrate", "Important Scale", "Non-Important Scale"]]

for _ in range(0, pass_encoding):
    channels_regular = []
    channels_adaptive = []

    for c in range(3):
        channel = image_ycc[:, :, c]
        reconstructed_regular = np.zeros_like(channel, dtype=np.uint8)
        reconstructed_adaptive = np.zeros_like(channel, dtype=np.uint8)
        regular_encoded_blocks = []
        adaptive_encoded_blocks = []

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = channel[i:i+8, j:j+8]
                if block.shape != (8, 8):  # Skip incomplete blocks
                    continue

                # Regular encoding
                quantized_block = apply_dct_quantization(block, jpeg_quantization_table)
                regular_encoded_blocks.append(quantized_block)
                reconstructed_regular[i:i+8, j:j+8] = inverse_dct_quantization(quantized_block, jpeg_quantization_table)

                # Adaptive encoding
                if blocks_with_edges[i//8, j//8]:
                    adaptive_quant_table = jpeg_quantization_table * important_scale
                else:
                    adaptive_quant_table = jpeg_quantization_table * non_important_scale
                
                quantized_block_adaptive = apply_dct_quantization(block, adaptive_quant_table)
                adaptive_encoded_blocks.append(quantized_block_adaptive)
                reconstructed_adaptive[i:i+8, j:j+8] = inverse_dct_quantization(quantized_block_adaptive, adaptive_quant_table)

        channels_regular.append(reconstructed_regular)
        channels_adaptive.append(reconstructed_adaptive)

    # Merge channels
    regular_encoded_image = cv2.merge(channels_regular)
    adaptive_encoded_image = cv2.merge(channels_adaptive)

    # Calculate bitrates
    regular_bitrate = calculate_bitrate(regular_encoded_blocks)
    adaptive_bitrate = calculate_bitrate(adaptive_encoded_blocks)
    

    # Adjust scales
    bitrate_difference = abs(regular_bitrate - adaptive_bitrate)
    
    csv_data.append([_ + 1, regular_bitrate, adaptive_bitrate, important_scale, non_important_scale])
    print(f"Pass {_ + 1}: Regular Bitrate = {regular_bitrate:.2f}, Adaptive Bitrate = {adaptive_bitrate:.2f}, Important Scale = {important_scale:.2f}, Non-Important Scale = {non_important_scale:.2f}")
            
    # Break loop if the difference is within tolerance
    if bitrate_difference <= threshold_tolerance:
        print("Threshold reached -> Bitrates Similar.")
        break
    
    if bitrate_difference > threshold_tolerance:
        if regular_bitrate > adaptive_bitrate:
            important_scale -= 0.05
        else:
            important_scale += 0.05
    
    
    
cv2.imwrite(os.path.join(output_folder, f"regular_encoded_image_pass.jpg"), cv2.cvtColor(regular_encoded_image, cv2.COLOR_YCrCb2BGR))
cv2.imwrite(os.path.join(output_folder, f"adaptive_encoded_image_pass.jpg"), cv2.cvtColor(adaptive_encoded_image, cv2.COLOR_YCrCb2BGR))
cv2.imwrite(os.path.join(output_folder, "edges_signal.jpg"), (blocks_with_edges.astype(np.uint8) * 255))

# Save CSV report
csv_path = os.path.join(output_folder, "bitrate_report.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"Images and report saved in folder: {output_folder}")

