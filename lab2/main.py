import cv2
import imutils
import os

file_name = os.path.join(os.path.dirname(__file__), 'img.jpg')
assert os.path.exists(file_name)
image = cv2.imread(file_name)
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Field", output)
cv2.waitKey(0)
