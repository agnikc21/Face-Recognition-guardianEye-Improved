import sys
import os
from database_manager import DatabaseManager
from face_detector import FaceDetector
from face_trainer import FaceTrainer
from face_recognizer import FaceRecognizer

class FaceRecognitionApp:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.face_detector = FaceDetector()
        self.face_trainer = FaceTrainer()
        self.face_recognizer = FaceRecognizer()
    
    def add_person(self):
        print("\n--- Add New Person ---")
        name = input("Enter name: ").strip()
        while not name:
            name = input("Name cannot be empty. Enter name: ").strip()
        
        try:
            age = int(input("Enter age: "))
        except ValueError:
            age = 0
        
        gender = input("Enter gender (M/F/Other): ").strip()
        notes = input("Enter notes (optional): ").strip()
        
        person_id = self.db_manager.get_next_id()
        print(f"Assigned ID: {person_id}")
        
        # Capture face data
        samples_captured = self.face_detector.capture_face_data(person_id)
        
        if samples_captured > 0:
            # Save to database
            self.db_manager.insert_person(person_id, name, age, gender, notes)
            print(f"Person {name} added successfully with ID: {person_id}")
            
            # Ask to retrain model
            retrain = input("Retrain model now? (y/n): ").lower().startswith('y')
            if retrain:
                self.train_model()
        else:
            print("No face data captured. Person not added.")
    
    def train_model(self):
        print("\n--- Training Model ---")
        success = self.face_trainer.train_model()
        if success:
            print("Model training completed successfully!")
            # Reload the recognizer
            self.face_recognizer = FaceRecognizer()
        else:
            print("Model training failed!")
    
    def start_recognition(self):
        print("\n--- Starting Face Recognition ---")
        self.face_recognizer.recognize_faces()
    
    def list_people(self):
        print("\n--- People in Database ---")
        people = self.db_manager.get_all_people()
        if not people:
            print("No people found in database.")
            return
        
        print(f"{'ID':<5} {'Name':<20} {'Age':<5} {'Gender':<8} {'Notes':<30}")
        print("-" * 70)
        for person in people:
            print(f"{person[0]:<5} {person[1]:<20} {person[2]:<5} {person[3]:<8} {person[4]:<30}")
    
    def delete_person(self):
        print("\n--- Delete Person ---")
        self.list_people()
        
        try:
            person_id = int(input("Enter ID of person to delete: "))
            person = self.db_manager.get_person(person_id)
            
            if person:
                confirm = input(f"Delete {person[1]} (ID: {person_id})? (y/n): ")
                if confirm.lower().startswith('y'):
                    self.db_manager.delete_person(person_id)
                    
                    # Delete associated image files
                    dataset_path = "data/dataset"
                    if os.path.exists(dataset_path):
                        for filename in os.listdir(dataset_path):
                            if filename.startswith(f"User.{person_id}."):
                                os.remove(os.path.join(dataset_path, filename))
                    
                    print(f"Person {person[1]} deleted successfully.")
                    
                    # Ask to retrain model
                    retrain = input("Retrain model now? (y/n): ").lower().startswith('y')
                    if retrain:
                        self.train_model()
            else:
                print("Person not found.")
        except ValueError:
            print("Invalid ID.")
    
    def show_menu(self):
        print("\n" + "="*50)
        print("         FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. Add New Person")
        print("2. Train Model")
        print("3. Start Face Recognition")
        print("4. List All People")
        print("5. Delete Person")
        print("6. Exit")
        print("="*50)
    
    def run(self):
        while True:
            self.show_menu()
            try:
                choice = input("Enter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.add_person()
                elif choice == '2':
                    self.train_model()
                elif choice == '3':
                    self.start_recognition()
                elif choice == '4':
                    self.list_people()
                elif choice == '5':
                    self.delete_person()
                elif choice == '6':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    FaceRecognitionApp().run()
