import cv2
import imutils
import numpy as np

screen_width = 1920
screen_height = 1080


image = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255



cv2.rectangle(image, (200, 200), (390, 390), (255, 0, 0), -1)  # Синий квадрат
cv2.rectangle(image, (600, 250), (1000, 380), (0, 165, 255), -1)  # Оранжевый прямоугольник

pts = np.array([[800, 400], [700, 700], [1000, 800]], np.int32)
cv2.fillPoly(image, [pts], (0, 0, 255))  # Желтый треугольник

pts = np.array([[1050, 820], [700, 1000], [1400, 1000], [1200, 820]], np.int32)
cv2.fillPoly(image, [pts], (128, 0, 128))  # Фиолетовая трапеция

cv2.circle(image, (1600, 500), 50, (0, 0, 255), -1)  # Красный круг
cv2.circle(image, (350, 700), 132, (255, 180, 100), -1)  # голубой круг BGR

pts = np.array([[1150, 500], [1250, 700], [1350, 500], [1250, 300]], np.int32)
cv2.fillPoly(image, [pts], (255, 0, 255))  # розовый ромб

pts = np.array([[800, 400], [700, 700], [1000, 800]], np.int32)
cv2.fillPoly(image, [pts], (255, 0, 255))  # Желтый треугольник

pts = np.array([[1350, 200], [1450, 200], [1400, 113]], np.int32)
cv2.fillPoly(image, [pts], (75, 0, 160)) # равносторонний треугольник

pts = np.array([[1500, 500], [1400, 500], [1400, 800], [1600, 800], [1700, 720]], np.int32)
cv2.fillPoly(image, [pts], (100, 255, 0))  # пятиугольник

pts = np.array([[300, 1000], [500, 1000], [690, 750]], np.int32)
cv2.fillPoly(image, [pts], (105, 40, 0))  # четырехугольник

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
# cv2.imshow("Thresh", thresh)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outlines = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outlines = imutils.grab_contours(outlines)

output = "Group 1245, Kuranov Grigory Sergeevich, program was able to found {} objects".format(len(outlines))
print(output)
cv2.drawContours(image, outlines, -1, (0, 0, 0), 3)
cv2.putText(image, output, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("", image)
cv2.waitKey(0)