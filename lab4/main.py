import cv2
import numpy as np

screen_width = 1920
screen_height = 1080

image = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

cv2.rectangle(image, (200, 200), (390, 390), (255, 0, 0), -1)  # Синий квадрат
cv2.rectangle(image, (600, 250), (1000, 380), (0, 165, 255), -1)  # Оранжевый прямоугольник

pts = np.array([[800, 400], [700, 700], [1000, 800]], np.int32)
cv2.fillPoly(image, [pts], (0, 0, 255))  # Красный треугольник

pts = np.array([[1040, 830], [710, 1010], [1390, 1010], [1210, 830]], np.int32)
cv2.fillPoly(image, [pts], (128, 0, 128))  # Фиолетовая трапеция

cv2.circle(image, (1600, 500), 50, (0, 0, 255), -1)  # Красный круг
cv2.circle(image, (350, 700), 132, (255, 180, 100), -1)  # Голубой круг

pts = np.array([[1150, 500], [1250, 700], [1350, 500], [1250, 300]], np.int32)
cv2.fillPoly(image, [pts], (255, 0, 255))  # Розовый ромб

pts = np.array([[800, 400], [700, 700], [1000, 800]], np.int32)
cv2.fillPoly(image, [pts], (255, 0, 255))  # Розовый треугольник

# 6 фигур копируем на второе пространство(хз как назвать)
annotated = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color_groups = {}
color_index = 1

for cnt in contours:
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    mean_color = cv2.mean(image, mask=mask)[:3]

    matched = False
    for color in color_groups:
        if all(abs(mean_color[i] - color[i]) < 20 for i in range(3)):
            color_name = color_groups[color]
            matched = True
            break

    if not matched:
        color_name = f'Color {color_index}'
        color_groups[tuple(mean_color)] = color_name
        color_index += 1

    area = cv2.contourArea(cnt)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        text = f"<{color_name}> ; <{area:.0f} px>"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.putText(annotated, text, (cX - text_w // 2, cY + text_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imshow("Original Image", image)
cv2.imshow("Analyzed Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()