import cv2
import imutils
import numpy as np

screen_width = 1920
screen_height = 1080
tolerance = 2  # в пикселях
stats = {
    'Triangle': {'total': 0, 'types': {}},
    'FourAngler': {'total': 0, 'types': {}},
    'Pentagon': 0,
    'Circle': 0,
    'Other': 0
}

image = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

def drawing_figures():
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

    pts = np.array([[1350, 200], [1450, 200], [1400, 113]], np.int32)
    cv2.fillPoly(image, [pts], (75, 0, 160))  # Темно-фиолетовый треугольник

    pts = np.array([[1500, 500], [1400, 500], [1400, 800], [1600, 800], [1700, 720]], np.int32)
    cv2.fillPoly(image, [pts], (100, 255, 0))  # Зеленый пятиугольник

    pts = np.array([[300, 1000], [500, 1000], [690, 750]], np.int32)
    cv2.fillPoly(image, [pts], (105, 40, 0))  # Коричневый четырехугольник

def is_parallel(v1, v2):
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) < tolerance

def detect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cnt = len(approx)
    shape = "-_-"
    coords = [[int(point[0][0]), int(point[0][1])] for point in approx]
    if cnt == 3:
        shape = "Triangle"
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        a = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        b = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5
        c = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5

        sides = sorted([a, b, c])

        if abs(a - b) < tolerance and abs(b - c) < tolerance:
            shape = "Ravnostoronnii Triangle"
        elif abs(a - b) < tolerance or abs(b - c) < tolerance or abs(a - c) < tolerance:
            shape = "Ravnobedrennii Triangle"
        else:
            if abs(sides[2] ** 2 - (sides[0] ** 2 + sides[1] ** 2)) < tolerance ** 2:
                shape = "Priamougolnii Triangle"
            else:
                shape = "Obichnii Triangle"

    elif cnt == 4:
        shape = "FourAngler"
        x = [p[0] for p in coords]
        y = [p[1] for p in coords]

        q_ab = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** 0.5
        q_bc = ((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2) ** 0.5
        q_cd = ((x[3] - x[2]) ** 2 + (y[3] - y[2]) ** 2) ** 0.5
        q_da = ((x[0] - x[3]) ** 2 + (y[0] - y[3]) ** 2) ** 0.5

        diag1 = ((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2) ** 0.5
        diag2 = ((x[3] - x[1]) ** 2 + (y[3] - y[1]) ** 2) ** 0.5

        if (abs(q_ab - q_bc) < tolerance and abs(q_bc - q_cd) < tolerance and abs(q_cd - q_da) < tolerance):
            if abs(diag1 - diag2) < tolerance:
                shape = "Square"
            else:
                shape = "Romb"
        elif (abs(q_ab - q_cd) < tolerance and
              abs(q_bc - q_da) < tolerance):
            if abs(diag1 - diag2) < tolerance:
                shape = "Rectangle"
            else:
                shape = "Parallelogram"
        else:
            vectors = [
                (x[1] - x[0], y[1] - y[0]),
                (x[2] - x[1], y[2] - y[1]),
                (x[3] - x[2], y[3] - y[2]),
                (x[0] - x[3], y[0] - y[3])
            ]

            parallel_pairs = 0
            if is_parallel(vectors[0], vectors[2]):
                parallel_pairs += 1
            if is_parallel(vectors[1], vectors[3]):
                parallel_pairs += 1

            if parallel_pairs == 1:
                shape = "Trapecia"

    elif cnt == 5:
        shape = "Pentagon"
    else:
        shape = "Circle"

    return shape

def update_stats(shape):
    if 'Triangle' in shape:
        stats['Triangle']['total'] += 1
        t_type = shape.split()[-1]
        stats['Triangle']['types'][t_type] = stats['Triangle']['types'].get(t_type, 0) + 1
    elif shape in ['Square', 'Rectangle', 'Romb', 'Parallelogram', 'Trapecia']:
        stats['FourAngler']['total'] += 1
        stats['FourAngler']['types'][shape] = stats['FourAngler']['types'].get(shape, 0) + 1
    elif shape in ['Pentagon', 'Circle']:
        stats[shape] += 1
    else:
        stats['Other'] += 1

def print_stats():
    print(f"\nОбщее количество фигур: {len(contours)}\n")

    # Треугольники
    tri = stats['Triangle']
    print(f"Треугольников – {tri['total']}")
    if tri['types']:
        print("  Типы:")
        for t, count in tri['types'].items():
            print(f"  - {t}: {count}")

    # Четырехугольники
    four = stats['FourAngler']
    print(f"\nЧетырехугольников – {four['total']}")
    if four['types']:
        print("  Типы:")
        for t, count in four['types'].items():
            print(f"  - {t}: {count}")

    # Остальные фигуры
    print("\nПрочие фигуры:")
    print(f"Пятиугольников: {stats['Pentagon']}")
    print(f"Кругов: {stats['Circle']}")
    if stats['Other'] > 0:
        print(f"Других фигур: {stats['Other']}")

drawing_figures()
original_image = image.copy()  # Сохраняем оригинальное изображение для определения цвета

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

color_map = {
    (255, 0, 0): 'Blue',
    (0, 165, 255): 'Orange',
    (0, 0, 255): 'Red',
    (128, 0, 128): 'Purple',
    (255, 0, 255): 'Pink',
    (75, 0, 160): 'Dark Purple',
    (100, 255, 0): 'Green',
    (105, 40, 0): 'Brown',
    (255, 180, 100): 'Light Blue'
}

detected_shapes = []

for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Определение цвета из оригинального изображения
    color = original_image[cY, cX].tolist()
    color_tuple = tuple(color)
    color_name = color_map.get(color_tuple, 'Unknown')

    # Вычисление площади
    area = cv2.contourArea(c)

    shape = detect(c)
    detected_shapes.append({'shape': shape, 'area': area, 'color': color_name})
    update_stats(shape)

    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

outlines = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outlines = imutils.grab_contours(outlines)

output = f"Group 1245, Kuranov Grigory Sergeevich, program was able to found {len(outlines)} objects"
print(output)

cv2.drawContours(image, outlines, -1, (0, 0, 0), 3)
cv2.putText(image, output, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
print_stats()

# Вывод площадей и цветов
print("\nПлощади всех найденных фигур:")
for idx, shape_info in enumerate(detected_shapes, 1):
    print(f"Фигура {idx}: {shape_info['area']:.2f}")

print("\nЦвета фигур:")
for idx, shape_info in enumerate(detected_shapes, 1):
    print(f"Фигура {idx}: {shape_info['color']}")

cv2.imshow("", image)
cv2.waitKey(0)