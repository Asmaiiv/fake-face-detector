# face_utils.py
from mtcnn import MTCNN
import cv2
import numpy as np

def detect_and_crop_face(image_path):
    """
    Detects the main face using MTCNN and crops it to 128x128.
    Raises ValueError if no face is found.
    """
    detector = MTCNN(min_face_size=50)
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        raise ValueError("فشل في تحميل الصورة.")
        
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img)

    if not detections:
        raise ValueError("لم يتم اكتشاف وجه في الصورة.")

    main_face = max(detections, key=lambda x: x['confidence'])

    x, y, w, h = main_face['box']
    padding = int(max(w, h) * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))

    return face
