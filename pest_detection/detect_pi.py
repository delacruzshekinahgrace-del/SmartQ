import cv2, time
from ultralytics import YOLO

MODEL = "/home/rpi/Documents/pest_detection/best_ncnn_model"
model = YOLO(MODEL)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.predict(frame, imgsz=320, conf=0.4, verbose=False)
    out = res[0].plot()
    fps = 1.0 / (time.time() - prev)
    prev = time.time()
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Maize Pest", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
