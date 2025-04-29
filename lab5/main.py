import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

image = cv2.imread('raw_imgs/img.png')

output_image = np.ones_like(image) * 255

tolerance = 2
color_map = {
    (255, 0, 0): 'Blue',
    (0, 165, 255): 'Orange',
    (0, 0, 255): 'Red',
    (128, 0, 128): 'Purple',
    (255, 0, 255): 'Pink',
    (75, 0, 160): 'Dark Blue',
    (100, 255, 0): 'Green',
    (105, 40, 0): 'Dark Blue',
    (255, 180, 100): 'Purple',
    (0, 255, 255): 'Yellow'
}


def refine_quadrilateral_type(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        return 'Quadrilateral'

    points = approx.reshape(-1, 2)

    sides = []
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        sides.append(np.linalg.norm([x2 - x1, y2 - y1]))

    angles = []
    for i in range(4):
        p0 = points[i]
        p1 = points[(i + 1) % 4]
        p2 = points[(i + 2) % 4]
        v1 = np.array(p0) - np.array(p1)
        v2 = np.array(p2) - np.array(p1)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5))
        angles.append(np.degrees(angle))

    right_angles = sum(1 for a in angles if 80 <= a <= 100)

    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

    def is_parallel(v1, v2, tol=5):
        angle = np.abs(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))
        angle = np.degrees(angle) % 180
        return min(angle, 180 - angle) < tol

    vectors = []
    for i in range(4):
        dx = points[(i + 1) % 4][0] - points[i][0]
        dy = points[(i + 1) % 4][1] - points[i][1]
        vectors.append((dx, dy))

    parallel_pairs = 0
    if is_parallel(vectors[0], vectors[2]):
        parallel_pairs += 1
    if is_parallel(vectors[1], vectors[3]):
        parallel_pairs += 1

    if right_angles == 4:
        return 'Square' if 0.95 <= aspect_ratio <= 1.05 else 'Rectangle'
    elif parallel_pairs >= 1:
        return 'Trapezoid'
    elif np.allclose(sides, [sides[0]] * 4, rtol=0.15):
        return 'Rhombus'
    else:
        return 'Quadrilateral'

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh


def get_features(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cnt = len(approx)
    area = cv2.contourArea(c)
    compactness = 4 * np.pi * area / (peri ** 2) if peri != 0 else 0
    return [cnt, compactness]


def train_knn():
    X = []
    y = []

    # Треугольники
    for _ in range(20):
        X.append([3, 0.5 + np.random.rand()*0.4])
        y.append('Triangle')

    # Четырехугольники
    for _ in range(20):
        X.append([4, 0.6 + np.random.rand()*0.3])
        y.append('FourAngler')

    # Круги
    for _ in range(20):
        X.append([8, 0.85 + np.random.rand()*0.15])
        y.append('Circle')

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

original = image.copy()


def process_image(img):
    global output_image
    output_image = np.ones_like(img) * 255
    thresh = preprocess_image(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    knn = train_knn()

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cnt = len(approx)
        area = cv2.contourArea(c)
        compactness = 4 * np.pi * area / (peri ** 2) if peri != 0 else 0

        shape = knn.predict([[cnt, compactness]])[0]

        if shape == 'FourAngler':
            shape = refine_quadrilateral_type(c)

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_color_bgr = cv2.mean(img, mask=mask)[:3]

        color_name = min(color_map.items(),
                         key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(mean_color_bgr)))[1]

        cv2.drawContours(output_image, [c], -1, (0, 0, 0), 2)
        cv2.fillPoly(output_image, [c], color=tuple(map(int, mean_color_bgr)))

        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            text = f"{shape} ({color_name})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            x = max(0, min(cX - tw // 2, img.shape[1] - tw))
            y = min(max(cY + th // 2, th), img.shape[0])

            cv2.putText(output_image, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output_image

processed = process_image(image.copy())

cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', processed)
cv2.waitKey(0)
cv2.destroyAllWindows()