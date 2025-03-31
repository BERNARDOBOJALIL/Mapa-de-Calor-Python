import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt") 


rtsp_url = "rtsp://Sistemas:liar3-c4m@172.18.140.29:554/stream1"

model = YOLO("yolov8n-seg.pt") 

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara TAPO")
    exit()
ret, frame = cap.read()
heatmap = np.zeros((1080, 1920), dtype=np.float32)  # Fijar el tamaño del heatmap a 1080x1920

alpha = 0.6
cooling_rate = 0.02 
heat_increase = 0.5  

cv2.namedWindow('Heatmap', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Heatmap', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame a 1080x1920
    frame_resized = cv2.resize(frame, (1920, 1080))

    results = model(frame_resized)[0]

    if results.masks is not None:  
        masks = results.masks.data.cpu().numpy()  
        classes = results.boxes.cls.cpu().numpy()  

        for i, mask in enumerate(masks):
            if int(classes[i]) == 0:  
                mask_resized = cv2.resize(mask, (1920, 1080))  
                heatmap += (mask_resized.astype(np.float32) * heat_increase)  

    heatmap = np.maximum(heatmap - (cooling_rate * heatmap), 0)

    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, frame_resized, 1 - alpha, 0)

    cv2.imshow('Heatmap', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
