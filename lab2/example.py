import cv2 as cv

img = cv.imread('img.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
masks = {
    "Желтый": ((25, 200, 200), (30, 255, 255)),
    "Зеленый": ((65, 150, 150), (70, 255, 255)),
    "Красный": ((175, 200, 200), (180, 255, 255)),
    "Синий": ((115, 150, 150), (120, 255, 255)),
    "Фиолетовый": ((145, 100, 100), (155, 255, 255)),
}
colors = ["Желтый", "Зеленый", "Красный", "Синий", "Фиолетовый"]
for color in colors:
    mask = cv.inRange(hsv, masks[color][0], masks[color][1])
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 10]
    max_area = '0'
    if contours:
        max_area = str(max([cv.contourArea(cnt) for cnt in contours]))
    print(color + ": " + max_area)

#  трешхолд - бинарное изображение. если яркий желтый -> применим блюр
# 1) генерация фигуры и контуров
# 2) поиск контуров
# 3) отправл контуров и классификация фигур
# 4) cv2.moment - почему центр, что за пересечения
# несколько типов фигур: несколько треугольников -> с помощью геометрии numpy померить отрезки и углы
#  как решить задание по распознаванию вида фигуры

