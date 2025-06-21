import cv2, os, torch, numpy as np, time
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from Silent_Face_Anti_Spoofing.test import test
from fer import FER

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
EMPLOYEE_DB = os.path.join(ROOT, "employees")
ATTENDANCE_LOG = os.path.join(ROOT, "attendance.csv")
os.makedirs(EMPLOYEE_DB, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FaceClassificationModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=500, embedding_dim=512):
        super().__init__()
        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        flat_features_size = 256 * 8 * 8
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features_size, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )
        # Final classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        # Forward pass through feature extractor and embedding
        x = self.features(x)
        emb = self.embedding(x)
        norm_emb = F.normalize(emb, p=2, dim=1)
        if return_embedding: 
            return norm_emb
        return self.classifier(emb), norm_emb

class FaceTripletModel(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=128):
        super().__init__()
        # CNN feature extractor (same as classification)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        flat_features_size = 256 * 8 * 8
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features_size, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        # Forward pass for triplet embedding
        x = self.features(x)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1)

class AgeGenderDetector:
    def __init__(self):
        # Load pre-trained age and gender models
        d = os.path.join(ROOT, "Age_Gender_Detection")
        self.ageNet = cv2.dnn.readNet(os.path.join(d, "age_net.caffemodel"), os.path.join(d, "age_deploy.prototxt"))
        self.genderNet = cv2.dnn.readNet(os.path.join(d, "gender_net.caffemodel"), os.path.join(d, "gender_deploy.prototxt"))
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
        self.genderList = ['Male','Female']
        self.mean = (78.426, 87.769, 114.896)

    def detect(self, face):
        # Predict age and gender from a face image
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.mean, swapRB=False)
        self.genderNet.setInput(blob)
        gender = self.genderList[self.genderNet.forward()[0].argmax()]
        self.ageNet.setInput(blob)
        age = self.ageList[self.ageNet.forward()[0].argmax()]
        return age, gender

class EmotionDetector:
    def __init__(self):
        # Initialize FER emotion detector
        self.detector = FER(mtcnn=True)

    def detect(self, face):
        # Predict emotion from a face image
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_emotions(rgb)
        if result and result[0]["emotions"]:
            em = max(result[0]["emotions"], key=result[0]["emotions"].get)
            return em.capitalize()
        return "Neutral"

class SpoofDetector:
    def __init__(self):
        # Set up anti-spoofing model directory
        self.model_dir = os.path.join(ROOT, "Silent_Face_Anti_Spoofing/resources/anti_spoof_models")

    def is_real(self, img, bbox):
        # Predict if the face is real or spoofed
        x, y, w, h = bbox
        face_img = img[y:y+h, x:x+w]
        height, width = face_img.shape[:2]
        target_width = int(height * 3/4)
        face_img = cv2.resize(face_img, (target_width, height))
        label = test(image=face_img, model_dir=self.model_dir, device_id=0)
        return label == 1

class FaceEngine:
    def __init__(self):
        # Initialize face detection, recognition, and auxiliary models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.classification_model = FaceClassificationModel().to(device)
        self.triplet_model = FaceTripletModel().to(device)
        # Load pre-trained weights for both models
        self.classification_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'classification_model.pth'), map_location=device))
        self.triplet_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'triplet_model.pth'), map_location=device))
        self.model_name = "classification"
        self.age_gender = AgeGenderDetector()
        self.emotion = EmotionDetector()
        self.spoof = SpoofDetector()
        self.load_db()

    def switch_model(self):
        # Switch between classification and triplet models for recognition
        self.model_name = "triplet" if self.model_name == "classification" else "classification"
        self.load_db()
        return self.model_name

    def load_db(self):
        # Load employee embeddings and info from the database
        self.ids, self.names, self.embeddings = [], [], []
        for emp_id in os.listdir(EMPLOYEE_DB):
            emp_dir = os.path.join(EMPLOYEE_DB, emp_id)
            info_path = os.path.join(emp_dir, "info.txt")
            name = emp_id
            if os.path.exists(info_path):
                with open(info_path) as f:
                    for line in f:
                        if line.startswith("Name:"):
                            name = line.split(":",1)[1].strip()
            emb_file = "class_embedding.npy" if self.model_name == "classification" else "triplet_embedding.npy"
            emb_path = os.path.join(emp_dir, emb_file)
            if os.path.exists(emb_path):
                self.ids.append(emp_id)
                self.names.append(name)
                self.embeddings.append(torch.tensor(np.load(emb_path)).to(device))

    def detect_faces(self, frame):
        # Detect faces in a frame using Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60,60))
        return [(frame[y:y+h, x:x+w], (x, y, w, h)) for (x, y, w, h) in faces]

    def get_embedding(self, face):
        # Get the embedding vector for a face using the selected model
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        t = transform(pil).unsqueeze(0).to(device)
        if self.model_name == "classification":
            self.classification_model.eval()
            with torch.no_grad():
                emb = self.classification_model(t, return_embedding=True).squeeze(0)
        else:
            self.triplet_model.eval()
            with torch.no_grad():
                emb = self.triplet_model(t).squeeze(0)
        return emb

    def recognize(self, emb):
        # Recognize the person by comparing embeddings
        if not self.embeddings: return None, None
        sims = [F.cosine_similarity(emb, e, dim=0).item() for e in self.embeddings]
        idx = np.argmax(sims)
        return (self.ids[idx], self.names[idx]) if sims[idx] > 0.5 else (None, None)

    def register(self, face, name):
        # Register a new employee: save face, embeddings, and info
        emp_id = f"emp_{int(time.time())}"
        emp_dir = os.path.join(EMPLOYEE_DB, emp_id)
        os.makedirs(emp_dir, exist_ok=True)
        cv2.imwrite(os.path.join(emp_dir, "face.jpg"), face)
        self.classification_model.eval()
        class_emb = self.get_embedding(face)
        np.save(os.path.join(emp_dir, "class_embedding.npy"), class_emb.cpu().numpy())
        self.model_name = "triplet"
        self.triplet_model.eval()
        triplet_emb = self.get_embedding(face)
        np.save(os.path.join(emp_dir, "triplet_embedding.npy"), triplet_emb.cpu().numpy())
        self.model_name = "classification"
        age, gender = self.age_gender.detect(face)
        emotion = self.emotion.detect(face)
        with open(os.path.join(emp_dir, "info.txt"), "w") as f:
            f.write(f"Name: {name}\nAge: {age}\nGender: {gender}\nRegistered: {datetime.now()}\nAnti-Spoof Check: Passed\nInitial Emotion: {emotion}\n")
        self.load_db()
        return emp_id

def log_attendance(emp_id, name):
    # Log attendance for an employee if not already marked today
    if not os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, 'w') as f: f.write("EmployeeID,Name,Date,Time,Status\n")
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_LOG, 'r') as f:
        for line in f:
            if emp_id in line and today in line:
                return False
    with open(ATTENDANCE_LOG, 'a') as f:
        now = datetime.now()
        f.write(f"{emp_id},{name},{now.date()},{now.time().strftime('%H:%M:%S')},Present\n")
    return True

class FaceApp:
    def __init__(self, root):
        # Initialize the main application, GUI, and camera
        self.root = root
        self.engine = FaceEngine()
        self.cap = cv2.VideoCapture(0)
        self.mode = "recognition"
        self.setup_ui()
        self.update()

    def setup_ui(self):
        # Set up the GUI components and buttons
        self.canvas = tk.Canvas(self.root, width=640, height=480); self.canvas.pack()
        self.btn_frame = tk.Frame(self.root); self.btn_frame.pack()
        self.btn_reg = tk.Button(self.btn_frame, text="Register", command=self.start_reg)
        self.btn_reg.pack(side=tk.LEFT)
        self.btn_switch = tk.Button(self.btn_frame, text="Switch Model", command=self.switch_model)
        self.btn_switch.pack(side=tk.LEFT)
        self.btn_capture = tk.Button(self.btn_frame, text="Capture", command=self.capture)
        self.btn_cancel = tk.Button(self.btn_frame, text="Cancel", command=self.cancel_reg)
        self.info = tk.Label(self.root, text="Ready"); self.info.pack()

    def switch_model(self):
        # Switch between classification and triplet models for recognition
        model = self.engine.switch_model()
        self.info.config(text=f"Switched to {model.title()} model")

    def start_reg(self):
        # Enter registration mode in the GUI
        self.mode = "registration"
        self.btn_reg.pack_forget(); self.btn_switch.pack_forget()
        self.btn_capture.pack(side=tk.LEFT); self.btn_cancel.pack(side=tk.LEFT)
        self.info.config(text="Registration: Position ONE real face and click Capture")

    def cancel_reg(self):
        # Cancel registration and return to recognition mode
        self.mode = "recognition"
        self.btn_capture.pack_forget(); self.btn_cancel.pack_forget()
        self.btn_reg.pack(side=tk.LEFT); self.btn_switch.pack(side=tk.LEFT)
        self.info.config(text="Ready")

    def capture(self):
        # Capture a face for registration, check anti-spoofing, and register if valid
        ret, frame = self.cap.read()
        faces = self.engine.detect_faces(frame)
        if len(faces) == 1 and self.engine.spoof.is_real(frame, faces[0][1]):
            emb = self.engine.get_embedding(faces[0][0])
            emp_id, name = self.engine.recognize(emb)
            if emp_id:
                messagebox.showinfo("Already Registered", f"This person is already inside the database, {emp_id}-{name}")
                self.cancel_reg()
                return
            name = simpledialog.askstring("Register", "Enter name:")
            if name:
                emp_id = self.engine.register(faces[0][0], name)
                messagebox.showinfo("Success", f"Registered {name} ({emp_id})")
                self.cancel_reg()
        else:
            messagebox.showerror("Error", "Need one real face for registration.")

    def draw_faces(self, frame, faces):
        # Draw bounding boxes and labels for detected faces, and log attendance
        for face, (x, y, w, h) in faces:
            is_real = self.engine.spoof.is_real(frame, (x, y, w, h))
            age, gender = self.engine.age_gender.detect(face)
            emotion = self.engine.emotion.detect(face)
            name, emp_id = "Unknown", None
            if is_real:
                emb = self.engine.get_embedding(face)
                emp_id, name = self.engine.recognize(emb)
                if emp_id:
                    if log_attendance(emp_id, name):
                        self.info.config(text=f"Attendance marked: {name}")
                    else:
                        self.info.config(text=f"Already marked: {name}")
            label = f"{name} | {age} | {gender} | {emotion} | {'Real' if is_real else 'Fake'}"
            color = (0,255,0) if is_real else (0,0,255)

            # Draw bounding box
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

            # Get text size
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # Draw filled rectangle as background for text
            cv2.rectangle(frame, (x, y-10-th), (x+tw, y-10+baseline), color, -1)
            # Draw text in black
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    def update(self):
        # Main loop: capture frame, detect faces, update GUI
        ret, frame = self.cap.read()
        if ret:
            faces = self.engine.detect_faces(frame)
            if self.mode == "recognition":
                self.draw_faces(frame, faces)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            self.canvas.image = img
        self.root.after(33, self.update)

def main():
    # Entry point: start the GUI application
    root = tk.Tk()
    root.title("Face Recognition")
    FaceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()