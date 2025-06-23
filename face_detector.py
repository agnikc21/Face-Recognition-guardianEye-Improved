import cv2
import numpy as np
import os
from pathlib import Path

class FaceDetector:
    def __init__(self, dataset_path="data/dataset"):
        self.dataset_path = dataset_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        os.makedirs(dataset_path, exist_ok=True)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def capture_face_data(self, person_id, samples_needed=150):
        cap = cv2.VideoCapture(0)
        sample_count = 0
        
        print(f"Capturing face data for person ID: {person_id}")
        print("Press 'q' to quit early or wait for automatic completion")
        
        while sample_count < samples_needed:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces, gray = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                sample_count += 1
                
                # Save face image
                face_img = gray[y:y+h, x:x+w]
                filename = f"{self.dataset_path}/User.{person_id}.{sample_count}.jpg"
                cv2.imwrite(filename, face_img)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {sample_count}/{samples_needed}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if sample_count >= samples_needed:
                    break
            
            cv2.imshow("Capturing Face Data", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {sample_count} samples for person ID: {person_id}")
        return sample_count
    
    def load_training_data(self):
        faces = []
        labels = []
        
        if not os.path.exists(self.dataset_path):
            return faces, labels
        
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.jpg') and filename.startswith('User.'):
                try:
                    parts = filename.split('.')
                    person_id = int(parts[1])
                    
                    img_path = os.path.join(self.dataset_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        faces.append(img)
                        labels.append(person_id)
                except (ValueError, IndexError):
                    continue
        
        return faces, labels
