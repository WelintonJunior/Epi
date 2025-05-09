import os
import cv2
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("A chave da API não foi encontrada. Crie um arquivo .env com API_KEY=xxxx")

dataset_path = "dataset/Hard-Hats-4"
dataset_yaml = os.path.join(dataset_path, "data.yaml")

if not os.path.exists(dataset_yaml):
    print("Baixando o dataset da Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("hard-hats-fhbh5")
    dataset = project.version(4).download("yolov8")
    print("Download concluído em:", dataset.location)
else:
    print("Dataset já existe em:", dataset_path)

best_model_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(best_model_path):
    print("Treinando o modelo YOLOv8n...")
    model = YOLO('yolov8n.pt')
    model.train(data=dataset_yaml, epochs=50, imgsz=416)
else:
    print("Modelo já treinado.")

model = YOLO(best_model_path)
model.to('cpu')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam.")

print("Iniciando detecção em tempo real. Pressione 'q' para sair.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da câmera.")
        break

    resized_frame = cv2.resize(frame, (416, 416))
    results = model.predict(source=resized_frame, conf=0.4, imgsz=416)
    annotated_frame = results[0].plot()

    cv2.imshow("Detecção de EPIs", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
