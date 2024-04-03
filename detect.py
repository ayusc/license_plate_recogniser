import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
import os

tesseract_path = os.popen('which tesseract').read().strip()
pytesseract.pytesseract.tesseract_cmd = tesseract_path

image = cv2.imread('image.jpg')

try:
 image = imutils.resize(image, width=500)
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 gray = cv2.bilateralFilter(gray, 11, 17, 17)
 edged = cv2.Canny(gray, 170, 200)
 (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
 cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]

 npcount = None 
 count = 0
 for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  
        npcount = approx 
        break

 mask = np.zeros(gray.shape,np.uint8)
 nimage = cv2.drawContours(mask,[npcount],0,255,-1)
 nimage = cv2.bitwise_and(image,image,mask=mask)
 config = ('-l eng --oem 1 --psm 3')
 text = pytesseract.image_to_string(nimage, config=config)
 raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[text]}
 df = pd.DataFrame(raw_data)
 df.to_csv('data.csv',mode='a')
 if text:
    print("\n"+"Found licence plate Number: "+text+"\n")
 else:
    print("\nSorry! I couldn't recognise a license plate in the provided image\n")
except Exception as e:
    print(f"\nSorry an error occurred !\nException Caught: {e}\n")
