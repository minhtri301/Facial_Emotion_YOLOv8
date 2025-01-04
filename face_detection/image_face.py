from ultralytics import YOLO
import cv2

face_model_path = 'trainModelAI/yolov8n-face.pt'
image_path = 'images/a2v.jpg'

face_model = YOLO(face_model_path)


img = cv2.imread(image_path)
if img is None:
    print("Không thể đọc ảnh!")

face_results = face_model.predict(img, conf=0.5, verbose=False)
face_result = face_results[0]

if not face_result.boxes:
    cv2.imshow("Emotion Detection", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

for box in face_result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_face = img[y1:y2, x1:x2]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Emotion Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

