import cv2
import numpy as np

screen_width = 1920
screen_height = 1080

image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
# image = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255


group_number = "1245"
name = "Kuranov Grigory"

cv2.rectangle(image, (200, 200), (390, 390), (255, 0, 0), -1)  # Синий квадрат
cv2.rectangle(image, (400, 150), (1000, 300), (0, 165, 255), -1)  # Оранжевый прямоугольник
pts = np.array([[800, 400], [700, 700], [1000, 800]], np.int32)
cv2.fillPoly(image, [pts], (0, 255, 255))  # Желтый треугольник
pts = np.array([[1200, 800], [1000, 1000], [1400, 1000], [1600, 800]], np.int32)
cv2.fillPoly(image, [pts], (128, 0, 128))  # Фиолетовая трапеция
cv2.circle(image, (1600, 500), 50, (0, 0, 255), -1)  # Красный круг

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
contour_count = f"{len(contours)}"

# info output
info = group_number + " " + name + ". Contours found: " + contour_count
cv2.putText(image, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
print(info)

cv2.imshow("Intellectual Systems. Lab1", image)
cv2.waitKey(0)