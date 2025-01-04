from ultralytics import YOLO
import cv2

face_model_path = '../trainModelAI/yolov8n-face.pt'
face_model = YOLO(face_model_path)

webcam_video = cv2.VideoCapture(0)

if not webcam_video.isOpened():
    print("Không thể mở webcam.")

while True:
    ret, frame = webcam_video.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam.")
        break

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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcam_video.release()
cv2.destroyAllWindows()

