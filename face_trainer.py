import cv2
import numpy as np
import os
from face_detector import FaceDetector

class FaceTrainer:
    def __init__(self, model_path="data/recognizer/trainingData.yml"):
        self.model_path = model_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = FaceDetector()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def train_model(self):
        print("Loading training data...")
        faces, labels = self.face_detector.load_training_data()
        
        if len(faces) == 0:
            print("No training data found. Please capture face data first.")
            return False
        
        print(f"Training with {len(faces)} face samples...")
        
        try:
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(self.model_path)
            print(f"Model trained and saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def update_model(self, new_faces, new_labels):
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.recognizer.update(new_faces, np.array(new_labels))
                self.recognizer.save(self.model_path)
                print("Model updated successfully")
                return True
            except Exception as e:
                print(f"Error updating model: {e}")
                return False
        else:
            return self.train_model()
