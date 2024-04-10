import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
import os

# Set Tesseract path
tesseract_path = os.popen('which tesseract').read().strip()
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Define function to process each image
def process_image(image_path):
    try:
        image = cv2.imread(image_path)

        # Resize image
        image = imutils.resize(image, width=500)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply Canny edge detection
        edged = cv2.Canny(gray, 170, 200)
        
        # Find contours
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        npcount = None 
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                npcount = approx 
                break

        # Create mask
        mask = np.zeros(gray.shape, np.uint8)
        nimage = cv2.drawContours(mask, [npcount], 0, 255, -1)
        nimage = cv2.bitwise_and(image, image, mask=mask)
        
        # Perform OCR
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(nimage, config=config)
        
        # Print license plate number if found
        if text:
            print("\nFound license plate Number in", image_path, ":", text)
        else:
            #print("\nSorry! Couldn't recognize a license plate in", image_path)
            pass
    except cv2.error:
        #print("\nSorry! Couldn't recognize a license plate in", image_path)
        pass 

# Directory containing images
image_dir = 'images'

# Create directory if not present
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Iterate over each image file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        #print("\nProcessing", image_path)
        process_image(image_path)
         
