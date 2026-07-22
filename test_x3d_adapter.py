import cv2
from x3d_adapter import X3DViolenceDetector, ClipBuffer

detector = X3DViolenceDetector()

# Load a test video (any RWF-2000 clip or your own test video works)
cap = cv2.VideoCapture("data/RWF-2000/train/NonFight/_q5Nwh4Z6ao_0.avi")
buffer = ClipBuffer(maxlen=32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    buffer.add(frame)

cap.release()

if buffer.is_ready():
    label, confidence = detector.predict(buffer.get_clip())
    print(f"Prediction: {label} ({confidence:.4f})")
else:
    print("Not enough frames collected.")