from ultralytics import YOLO
import cv2

face_model_path = 'trainModelAI/yolov8n-face.pt'
emotion_model_path = 'trainModelAI/yolov8s-cls/best_CLS2.pt'

face_model = YOLO(face_model_path)
emotion_model = YOLO(emotion_model_path)

def FER_realtime(face_model, emotion_model):
    webcam_video = cv2.VideoCapture(0)

    if not webcam_video.isOpened():
        print("Không thể mở webcam.")
        return

    while True:
        ret, frame = webcam_video.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam.")
            break

        # Dự đoán khuôn mặt
        face_results = face_model.predict(frame, conf=0.5, verbose=False)
        face_result = face_results[0]

        if not face_result.boxes:
            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        for box in face_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                continue

            resized_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

            # Dự đoán cảm xúc
            emotion_result = emotion_model.predict(resized_face_rgb, conf=0.5, verbose=False)
            probs = emotion_result[0].probs  # Lấy đối tượng Probs

            if probs is not None:
                cls_id = probs.top1  # Lấy lớp có xác suất cao nhất
                emotion_label = emotion_result[0].names[cls_id]  # Tên cảm xúc
            else:
                emotion_label = "Unknown"

            # Vẽ bounding box và nhãn cảm xúc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    webcam_video.release()
    cv2.destroyAllWindows()

FER_realtime(face_model, emotion_model)