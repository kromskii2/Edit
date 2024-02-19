import cv2
import os
from datetime import datetime
from PIL import Image
import imagehash

# Инициализация классификатора для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Запуск видеопотока с веб-камеры
video_capture = cv2.VideoCapture(0)

# Создание папки для сохранения лиц, если она не существует
if not os.path.exists('faces'):
    os.makedirs('faces')

# Список хэшей сохраненных лиц
saved_faces_hashes = []


def calculate_image_hash(image):
    return imagehash.average_hash(Image.fromarray(image))


def is_similar_hash(hash1, hash2, tolerance=5):
    return hash1 - hash2 < tolerance


while True:
    # Считывание кадра из видеопотока
    ret, frame = video_capture.read()

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Сохранение каждого нового обнаруженного лица
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y + h, x:x + w]
        face_hash = calculate_image_hash(face_img)

        # Проверка на схожесть с уже сохраненными лицами
        similar_face = False
        for saved_hash in saved_faces_hashes:
            if is_similar_hash(face_hash, saved_hash):
                similar_face = True
                break

        if not similar_face:
            # Сохранение лица
            face_file_path = os.path.join('faces', f'face_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}.jpg')
            cv2.imwrite(face_file_path, face_img)
            saved_faces_hashes.append(face_hash)

            # Рисование прямоугольника вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение кадра с выделенными лицами
    cv2.imshow('Video', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
