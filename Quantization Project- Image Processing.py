#!/usr/bin/env python
# coding: utf-8

# In[1]:


#run in Jupyter notebook on HPC
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


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


# In[3]:


def apply_dct_quantization(block, quant_table):
    """Apply DCT and quantization to an 8x8 block."""
    dct_block = cv2.dct(block.astype(np.float32) - 128)
    quantized_block = np.round(dct_block / quant_table).astype(np.int32)
    return quantized_block


# In[4]:


def inverse_dct_quantization(block, quant_table):
    """Apply inverse DCT and de-quantization to an 8x8 block."""
    dequantized_block = block * quant_table
    idct_block = cv2.idct(dequantized_block.astype(np.float32)) + 128
    return np.clip(idct_block, 0, 255).astype(np.uint8)


# In[5]:


def calculate_bitrate(quantized_blocks):
    """Calculate bitrate based on the number of non-zero coefficients."""
    total_coefficients = sum(block.size for block in quantized_blocks)
    non_zero_coefficients = sum(np.count_nonzero(block) for block in quantized_blocks)
    return non_zero_coefficients / total_coefficients


# In[6]:


image = cv2.imread("/mnt/koko-storage/mlab/fcm/datasets/fcm_testdata/SFU_HW_Obj/ParkScene_1920x1080_24_val/images/ParkScene_1920x1080_24_val_000.png")
image_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
h, w, _ = image_ycc.shape

edges = cv2.Canny(image_ycc[:, :, 0], 100, 200)

blocks_with_edges = np.zeros((h // 8, w // 8), dtype=bool)
for i in range(0, h, 8):
    for j in range(0, w, 8):
        block = edges[i:i+8, j:j+8]
        if np.any(block):  # Mark block if edges are present
            blocks_with_edges[i//8, j//8] = True


# In[14]:


# Apply DCT + Quantization to each channel
channels_regular = []
channels_adaptive = []

# Parameters for iterative threshold adjustment
important_scale = 1.0  # Initial quality scale for important blocks
non_important_scale = 2.0  # Initial quality scale for non-important blocks
pass_encoding = 3  # Number of passes for encoding
threshold_tolerance = 0.01  # Tolerance for bitrate difference

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
                if blocks_with_edges[i//8, j//8]:  # Important block
                    adaptive_quant_table = jpeg_quantization_table * important_scale
                else:  # Non-important block
                    adaptive_quant_table = jpeg_quantization_table * non_important_scale
                
                quantized_block_adaptive = apply_dct_quantization(block, adaptive_quant_table)
                adaptive_encoded_blocks.append(quantized_block_adaptive)
                reconstructed_adaptive[i:i+8, j:j+8] = inverse_dct_quantization(quantized_block_adaptive, adaptive_quant_table)

        channels_regular.append(reconstructed_regular)
        channels_adaptive.append(reconstructed_adaptive)

    # Merge the regular and adaptive channels
    regular_encoded_image = cv2.merge(channels_regular)
    adaptive_encoded_image = cv2.merge(channels_adaptive)

    # Calculate bitrates
    regular_bitrate = calculate_bitrate(regular_encoded_blocks)
    adaptive_bitrate = calculate_bitrate(adaptive_encoded_blocks)

    # Adjust scales to minimize bitrate difference
    bitrate_difference = abs(regular_bitrate - adaptive_bitrate)

    if bitrate_difference > threshold_tolerance:
        if regular_bitrate > adaptive_bitrate:
            important_scale -= 0.05  # # Reduce compression (increase quality)
            
        else:
            important_scale += 0.05  # Decrease compression (improve quality)
            

    print(f"Pass {_ + 1}: Regular Bitrate = {regular_bitrate:.2f}, Adaptive Bitrate = {adaptive_bitrate:.2f}, "
          f"Important Scale = {important_scale:.2f}, Non-Important Scale = {non_important_scale:.2f}")

    # Break loop if the difference is within tolerance
    if bitrate_difference <= threshold_tolerance:
        print("Threshold reached! Bitrates are close enough.")
        break

# Convert YCrCb to BGR for display
regular_encoded_image_bgr = cv2.cvtColor(regular_encoded_image, cv2.COLOR_YCrCb2BGR)
adaptive_encoded_image_bgr = cv2.cvtColor(adaptive_encoded_image, cv2.COLOR_YCrCb2BGR)

# Save output images
cv2.imwrite("regular_encoded_image.jpg", regular_encoded_image_bgr)
cv2.imwrite("adaptive_encoded_image.jpg", adaptive_encoded_image_bgr)

plt.figure(figsize=(15, 10))


# In[8]:


# Original Image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[9]:


# Edges
plt.subplot(2, 3, 2)
plt.title("Edges/Contours")
plt.imshow(edges, cmap='gray')


# In[10]:


# Reconstructed Regular Encoding
plt.subplot(2, 3, 3)
plt.title(f"Regular Encoding\nBitrate: {regular_bitrate:.2f}")
plt.imshow(cv2.cvtColor(regular_encoded_image_bgr, cv2.COLOR_BGR2RGB))


# In[11]:


# Reconstructed Adaptive Encoding
plt.subplot(2, 3, 4)
plt.title(f"Adaptive Encoding\nBitrate: {adaptive_bitrate:.2f}")
plt.imshow(cv2.cvtColor(adaptive_encoded_image_bgr, cv2.COLOR_BGR2RGB))


# In[12]:


# Blocks with edges
plt.subplot(2, 3, 5)
plt.title("Blocks with Edges")
plt.imshow(blocks_with_edges, cmap='gray')


# In[13]:


plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:



