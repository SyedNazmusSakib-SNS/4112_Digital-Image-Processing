import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folders
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/step1_original', exist_ok=True)
os.makedirs('outputs/step2_histograms', exist_ok=True)
os.makedirs('outputs/step3_equalized', exist_ok=True)
os.makedirs('outputs/step4_clahe', exist_ok=True)
os.makedirs('outputs/step5_thresholding', exist_ok=True)
os.makedirs('outputs/step6_adaptive', exist_ok=True)

print("="*70)
print("LAB TASK 1: IMAGE ENHANCEMENT & THRESHOLDING")
print("="*70)

# Load images
img1 = cv2.imread('image_11.png')
img2 = cv2.imread('image_22.png')
img3 = cv2.imread('image_33.png')
img4 = cv2.imread('image_44.png')

print("\n✅ Images loaded successfully!")
print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")
print(f"Image 3 shape: {img3.shape}")
print(f"Image 4 shape: {img4.shape}")

# Helper function to resize images to same height
def resize_to_same_height(images, target_height=None):
    """Resize all images to the same height while maintaining aspect ratio"""
    if target_height is None:
        # Use minimum height among all images
        target_height = min(img.shape[0] for img in images)
    
    resized = []
    for img in images:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        resized.append(cv2.resize(img, (new_width, target_height)))
    return resized

# ============================================================================
# STEP 1: DISPLAY ORIGINAL IMAGES
# ============================================================================
print("\n" + "="*70)
print("STEP 1: ORIGINAL IMAGES")
print("="*70)

# Resize images to same height for concatenation
resized_imgs = resize_to_same_height([img1, img2, img3, img4])

# Display side by side
images_concat = np.concatenate(resized_imgs, axis=1)
cv2.imwrite('outputs/step1_original/all_images_color.jpg', images_concat)
print("✅ Saved: all_images_color.jpg")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

# Resize grayscale images
resized_grays = resize_to_same_height([gray1, gray2, gray3, gray4])

gray_concat = np.concatenate(resized_grays, axis=1)
cv2.imwrite('outputs/step1_original/all_images_gray.jpg', gray_concat)
print("✅ Saved: all_images_gray.jpg")

# ============================================================================
# STEP 2: CALCULATE AND PLOT ORIGINAL HISTOGRAMS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: ORIGINAL HISTOGRAMS")
print("="*70)

hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([gray3], [0], None, [256], [0, 256])
hist4 = cv2.calcHist([gray4], [0], None, [256], [0, 256])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Original Histograms', fontsize=16)

axes[0, 0].plot(hist1, color='blue')
axes[0, 0].set_title('Image 1')
axes[0, 0].set_xlabel('Pixel Intensity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(hist2, color='green')
axes[0, 1].set_title('Image 2')
axes[0, 1].set_xlabel('Pixel Intensity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(hist3, color='red')
axes[1, 0].set_title('Image 3')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(hist4, color='purple')
axes[1, 1].set_title('Image 4')
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/step2_histograms/original_histograms.png', dpi=150)
plt.close()
print("✅ Saved: original_histograms.png")

# ============================================================================
# STEP 3: HISTOGRAM EQUALIZATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3: HISTOGRAM EQUALIZATION")
print("="*70)

eq1 = cv2.equalizeHist(gray1)
eq2 = cv2.equalizeHist(gray2)
eq3 = cv2.equalizeHist(gray3)
eq4 = cv2.equalizeHist(gray4)

# Resize equalized images
resized_eq = resize_to_same_height([eq1, eq2, eq3, eq4])

# Save equalized images
eq_concat = np.concatenate(resized_eq, axis=1)
cv2.imwrite('outputs/step3_equalized/equalized_images.jpg', eq_concat)
print("✅ Saved: equalized_images.jpg")

# Compare before/after for each image
for i, (gray, eq, name) in enumerate([(gray1, eq1, 'img1'), (gray2, eq2, 'img2'), 
                                       (gray3, eq3, 'img3'), (gray4, eq4, 'img4')]):
    comparison = np.concatenate((gray, eq), axis=1)
    cv2.imwrite(f'outputs/step3_equalized/compare_{name}.jpg', comparison)

print("✅ Saved: Individual comparisons")

# Plot equalized histograms
hist_eq1 = cv2.calcHist([eq1], [0], None, [256], [0, 256])
hist_eq2 = cv2.calcHist([eq2], [0], None, [256], [0, 256])
hist_eq3 = cv2.calcHist([eq3], [0], None, [256], [0, 256])
hist_eq4 = cv2.calcHist([eq4], [0], None, [256], [0, 256])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Equalized Histograms', fontsize=16)

axes[0, 0].plot(hist_eq1, color='blue')
axes[0, 0].set_title('Image 1 - Equalized')
axes[0, 0].set_xlabel('Pixel Intensity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(hist_eq2, color='green')
axes[0, 1].set_title('Image 2 - Equalized')
axes[0, 1].set_xlabel('Pixel Intensity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(hist_eq3, color='red')
axes[1, 0].set_title('Image 3 - Equalized')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(hist_eq4, color='purple')
axes[1, 1].set_title('Image 4 - Equalized')
axes[1, 1].set_xlabel('Pixel Intensity')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/step3_equalized/equalized_histograms.png', dpi=150)
plt.close()
print("✅ Saved: equalized_histograms.png")

# ============================================================================
# STEP 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================
print("\n" + "="*70)
print("STEP 4: CLAHE ENHANCEMENT")
print("="*70)

clahe = cv2.createCLAHE(clipLimit=40)
clahe1 = clahe.apply(eq1)
clahe2 = clahe.apply(eq2)
clahe3 = clahe.apply(eq3)
clahe4 = clahe.apply(eq4)

# Resize CLAHE images
resized_clahe = resize_to_same_height([clahe1, clahe2, clahe3, clahe4])

# Save CLAHE images
clahe_concat = np.concatenate(resized_clahe, axis=1)
cv2.imwrite('outputs/step4_clahe/clahe_images.jpg', clahe_concat)
print("✅ Saved: clahe_images.jpg")

# Compare: Original -> Equalized -> CLAHE
for i, (gray, eq, clahe_img, name) in enumerate([(gray1, eq1, clahe1, 'img1'), 
                                                   (gray2, eq2, clahe2, 'img2'),
                                                   (gray3, eq3, clahe3, 'img3'), 
                                                   (gray4, eq4, clahe4, 'img4')]):
    comparison = np.concatenate((gray, eq, clahe_img), axis=1)
    cv2.imwrite(f'outputs/step4_clahe/progression_{name}.jpg', comparison)

print("✅ Saved: Progression comparisons (Original->Equalized->CLAHE)")

# ============================================================================
# STEP 5: THRESHOLDING OPERATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: THRESHOLDING OPERATIONS")
print("="*70)

th = 80
max_val = 255

# Apply all threshold types on image 1
ret, th1 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_BINARY_INV)
ret, th3 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_TOZERO)
ret, th4 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_TOZERO_INV)
ret, th5 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_TRUNC)
ret, th6 = cv2.threshold(clahe1, th, max_val, cv2.THRESH_OTSU)

# Add labels
cv2.putText(th1, "Binary", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.putText(th2, "Binary_Inv", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.putText(th3, "ToZero", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.putText(th4, "ToZero_Inv", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.putText(th5, "Trunc", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.putText(th6, "Otsu", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

# Save threshold results
row1 = np.concatenate((th1, th2, th3), axis=1)
row2 = np.concatenate((th4, th5, th6), axis=1)
cv2.imwrite('outputs/step5_thresholding/thresh_row1.jpg', row1)
cv2.imwrite('outputs/step5_thresholding/thresh_row2.jpg', row2)
print("✅ Saved: Thresholding results")

# ============================================================================
# STEP 6: ADAPTIVE THRESHOLDING
# ============================================================================
print("\n" + "="*70)
print("STEP 6: ADAPTIVE THRESHOLDING")
print("="*70)

# Apply adaptive thresholding on all images
adapt1_1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
adapt1_2 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 31, 3)
adapt1_3 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 13, 5)
adapt1_4 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 31, 4)

adapt2_1 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
adapt2_2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 31, 5)
adapt2_3 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 21, 5)
adapt2_4 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 31, 5)

# Save adaptive threshold results
adapt_row1 = np.concatenate((adapt1_1, adapt1_2, adapt1_3, adapt1_4), axis=1)
adapt_row2 = np.concatenate((adapt2_1, adapt2_2, adapt2_3, adapt2_4), axis=1)
cv2.imwrite('outputs/step6_adaptive/adaptive_img1.jpg', adapt_row1)
cv2.imwrite('outputs/step6_adaptive/adaptive_img2.jpg', adapt_row2)
print("✅ Saved: Adaptive thresholding results")

# ============================================================================
# STEP 7: OTSU THRESHOLDING ON ALL IMAGES
# ============================================================================
print("\n" + "="*70)
print("STEP 7: OTSU THRESHOLDING")
print("="*70)

ret1, otsu1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret2, otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret3, otsu3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret4, otsu4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Resize Otsu images
resized_otsu = resize_to_same_height([otsu1, otsu2, otsu3, otsu4])

otsu_concat = np.concatenate(resized_otsu, axis=1)
cv2.imwrite('outputs/step6_adaptive/otsu_all_images.jpg', otsu_concat)
print(f"✅ Saved: Otsu thresholding (thresholds: {ret1:.0f}, {ret2:.0f}, {ret3:.0f}, {ret4:.0f})")

print("\n" + "="*70)
print("✅ LAB TASK COMPLETE!")
print("="*70)
print("\nAll outputs saved in 'outputs/' folder:")
print("  📁 step1_original/  - Original color and grayscale images")
print("  📁 step2_histograms/ - Histogram plots")
print("  📁 step3_equalized/ - Histogram equalization results")
print("  📁 step4_clahe/     - CLAHE enhancement results")
print("  📁 step5_thresholding/ - Various thresholding methods")
print("  📁 step6_adaptive/  - Adaptive and Otsu thresholding")
print("="*70)