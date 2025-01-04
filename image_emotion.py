from ultralytics import YOLO
import cv2

face_model_path = 'trainModelAI/yolov8n-face.pt'
emotion_model_path = 'trainModelAI/yolov8s-cls/best_CLS2.pt'
image_path = 'images/a2v.jpg'

face_model = YOLO(face_model_path)
emotion_model = YOLO(emotion_model_path)

def FER_image(face_model, emotion_model, image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return

    face_results = face_model.predict(img, conf=0.5, verbose=False)
    face_result = face_results[0]

    if not face_result.boxes:
        cv2.imshow("Emotion Detection", img)
        if cv2.waitKey(0) == ord('q'):
            return

    for box in face_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_face = img[y1:y2, x1:x2]

        if cropped_face.size == 0:
            continue

        resized_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        emotion_result = emotion_model.predict(resized_face_rgb, conf=0.5, verbose=False)
        probs = emotion_result[0].probs  # Lấy đối tượng Probs

        if probs is not None:
            cls_id = probs.top1  # Lấy lớp có xác suất cao nhất
            emotion_label = emotion_result[0].names[cls_id]  # Tên cảm xúc
        else:
            emotion_label = "Unknown"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

FER_image(face_model, emotion_model, image_path)