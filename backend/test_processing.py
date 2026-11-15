
import cv2
from image_processing import enhance_image

# Load a test image
image = cv2.imread("test-image-converted.jpg")
if image is None:
    raise FileNotFoundError("Place test-image.jpg inside backend folder.")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply enhancement
enhanced, method = enhance_image(image, "Automatic")

# Convert RGB to BGR before saving
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

cv2.imwrite("enhanced_image.jpg", enhanced)

print("Enhancement successful!")
print("Selected method:", method)
print("Original dimensions:", image.shape)
print("Enhanced dimensions:", enhanced.shape)
