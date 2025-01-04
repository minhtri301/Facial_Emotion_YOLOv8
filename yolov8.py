from ultralytics import YOLO
import cv2

face_model_path = 'trainModelAI/yolov8n-face.pt'
emotion_model_path = 'trainModelAI/yolov8n/best.pt'
image_path = 'images/a2v.jpg'
face_model = YOLO(face_model_path)
emotion_model = YOLO(emotion_model_path)

def detect_and_classify_emotions(face_model, emotion_model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return

    face_results = face_model.predict(img)
    face_result = face_results[0]

    if not face_result.boxes:
        print("Không phát hiện khuôn mặt nào.")
        return

    for box in face_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_face = img[y1:y2, x1:x2]

        resized_face = cv2.resize(cropped_face, (320, 320))
        resized_face_rgb = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        emotion_results = emotion_model.predict(resized_face_rgb,imgsz=320)
        emotion_result = emotion_results[0]

        if not emotion_result.boxes:
            emotion_label = "Unknown"
            return
        for emo_box in emotion_result.boxes:
            cls_id = int(emo_box.cls[0])
            emotion_label = emotion_result.names[cls_id]


        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_and_classify_emotions(face_model, emotion_model, image_path)