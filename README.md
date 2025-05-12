
---

### 📁 `README.md`

````markdown
# 🧠 EPI Detector com YOLOv8 + Roboflow

Este projeto detecta EPIs (como capacetes) em tempo real usando YOLOv8 e um dataset da Roboflow.

---

## 🚀 Como executar

Siga os passos abaixo para rodar o projeto em qualquer máquina com Python instalado.

### 1. Clone o repositório

```bash
git clone https://github.com/WelintonJunior/Epi.git
cd Epi


### 2. (Opcional) Crie um ambiente virtual

```bash
python -m venv venv
# Ative o ambiente:
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Crie o arquivo `.env` com sua API KEY da Roboflow

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```
API_KEY=sua_chave_da_roboflow
```

### 5. Execute o projeto

```bash
python app.py
```

O sistema vai:

* Baixar o dataset da Roboflow (na primeira vez)
* Treinar o modelo YOLOv8 (na primeira vez)
* Abrir a webcam e iniciar a detecção de EPIs em tempo real

---

## 📦 Estrutura esperada

```
.
├── app.py
├── .env
├── requirements.txt
├── dataset/
└── runs/                # Criado automaticamente após o treino
```

---

## ⚠️ Observações

* O `.env` **NÃO** deve ser enviado para o GitHub (adicione ao `.gitignore`).
* O modelo é salvo após o primeiro treino e reutilizado nas próximas execuções.
* Pressione **`q`** para encerrar a detecção ao vivo.

---

## ✅ Requisitos

* Python 3.8+
* Webcam funcional

```

Para instalar o .exe, basta executar esse comando na raiz do projeto:

pyinstaller app.spec

Caso não tenha o pyinstaller instalado use esse comando:

pip install pyinstaller

