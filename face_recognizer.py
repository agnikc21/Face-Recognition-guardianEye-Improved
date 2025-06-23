import cv2
import os
from face_detector import FaceDetector
from database_manager import DatabaseManager

class FaceRecognizer:
    def __init__(self, model_path="data/recognizer/trainingData.yml", confidence_threshold=70):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = FaceDetector()
        self.db_manager = DatabaseManager()
        
        if os.path.exists(model_path):
            self.recognizer.read(model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("No trained model found. Please train the model first.")
    
    def recognize_faces(self):
        if not self.model_loaded:
            print("Model not loaded. Cannot perform recognition.")
            return
        
        cap = cv2.VideoCapture(0)
        print("Face recognition started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces, gray = self.face_detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                person_id, confidence = self.recognizer.predict(face_roi)
                
                if confidence < self.confidence_threshold:
                    person_data = self.db_manager.get_person(person_id)
                    if person_data:
                        name = person_data[1]  
                        age = person_data[2]   
                        gender = person_data[3] 
                        
                        # Draw green rectangle for recognized person
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display person info
                        y_offset = y + h + 20
                        cv2.putText(frame, f"Name: {name}", (x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Age: {age}", (x, y_offset + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gender: {gender}", (x, y_offset + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y_offset + 75), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, f"ID: {person_id} (No data)", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Draw red rectangle for unknown person
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown Person", (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}", (x, y+h+45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def recognize_single_frame(self, frame):
        if not self.model_loaded:
            return []
        
        faces, gray = self.face_detector.detect_faces(frame)
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            person_id, confidence = self.recognizer.predict(face_roi)
            
            result = {
                'bbox': (x, y, w, h),
                'person_id': person_id,
                'confidence': confidence,
                'recognized': confidence < self.confidence_threshold
            }
            
            if result['recognized']:
                person_data = self.db_manager.get_person(person_id)
                if person_data:
                    result['name'] = person_data[1]
                    result['age'] = person_data[2]
                    result['gender'] = person_data[3]
            
            recognized_faces.append(result)
        
        return recognized_faces
