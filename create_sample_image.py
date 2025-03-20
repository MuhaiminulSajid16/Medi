import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_prescription_image(output_path="sample_prescription.jpg"):
    """Create a sample prescription image for testing."""
    # Create a white image
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a font that looks like handwriting
    try:
        # Try to use a system font
        font_path = "C:\\Windows\\Fonts\\comic.ttf"  # Comic Sans as fallback
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, 24)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Add a header
    draw.text((50, 50), "Dr. Smith Medical Clinic", fill='black', font=font)
    draw.text((50, 90), "123 Health Street, Medical City", fill='black', font=font)
    draw.text((50, 130), "Phone: (123) 456-7890", fill='black', font=font)
    
    # Add a line
    draw.line([(50, 180), (width-50, 180)], fill='black', width=2)
    
    # Add patient information
    draw.text((50, 200), "Patient: John Doe", fill='black', font=font)
    draw.text((50, 240), "Date: 2023-10-15", fill='black', font=font)
    
    # Add prescription details
    draw.text((50, 300), "Rx:", fill='black', font=font)
    draw.text((100, 350), "Amoxicillin 500mg", fill='black', font=font)
    draw.text((100, 390), "Take 1 capsule three times daily for 7 days", fill='black', font=font)
    
    draw.text((100, 450), "Metformin 1000mg", fill='black', font=font)
    draw.text((100, 490), "Take 1 tablet twice daily with meals", fill='black', font=font)
    
    # Add signature
    draw.text((50, 600), "Signature: ___________________", fill='black', font=font)
    
    # Add some noise to make it look more realistic
    image_array = np.array(image)
    noise = np.random.normal(0, 5, image_array.shape).astype(np.uint8)
    noisy_image = cv2.add(image_array, noise)
    
    # Add slight blur to simulate scanning
    blurred_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
    
    # Save the image
    cv2.imwrite(output_path, blurred_image)
    print(f"Sample prescription image created at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_prescription_image() 