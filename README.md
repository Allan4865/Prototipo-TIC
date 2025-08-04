# PROTOTIPO DTIC  
Sistema de detección de objetos con apoyo de profundidad y TTS  
*Ayuda accesible en tiempo real para personas con discapacidad visual, basada en hardware convencional.*

---

## Tabla de contenidos
1. [Descripción](#descripción)  
2. [Características](#características)  
3. [Arquitectura](#arquitectura)  
4. [Requisitos](#requisitos)  
5. [Instalación](#instalación)  
---

## Descripción
El prototipo reconoce **10 objetos domésticos** (teclado, ratón, teléfono, etc.) y anuncia su proximidad con síntesis de voz en español.  
Combina:

* **YOLOv8-nano** – detección de objetos.  
* **MiDaS-small** – estimación de profundidad (_near / mid / far_).  
* **eSpeak-NG** – TTS offline.  

Todo se ejecuta localmente; no se envían datos a la nube.

---

## Características
- Detección en tiempo real **solo CPU**.  
- Clasificación de distancia (near/mid/far).  
- TTS en español con _mute/unmute_.  
- Interfaz OpenCV con overlay (cajas, texto, FPS).  
- Umbrales configurables de confianza y proximidad.  

---

## Arquitectura
![Diagrama de arquitectura](https://firebasestorage.googleapis.com/v0/b/correos-masivos-1c0c7.appspot.com/o/ChatGPT%20Image%203%20ago%202025%2C%2010_35_17%20p.m..png?alt=media&token=ec2cf050-0651-4305-b81f-8d84ee232d3a)


1. **Front-end** (ventana OpenCV) captura vídeo y muestra resultados.  
2. **API de inferencia** (YOLOv8-nano ONNX) devuelve objetos y confianza.  
3. **API de TTS** genera el audio que reproduce el front-end.  

---

## Requisitos

| Tipo            | Versión mínima | Notas                              |
| --------------- | ------------- | ---------------------------------- |
| Python          | 3.9           | 64 bit recomendado                 |
| OpenCV-Python   | 4.7           | `opencv-contrib` opcional          |
| Ultralytics     | 8.x           | Incluye PyTorch                    |
| onnxruntime     | 1.16          | Aceleración en CPU                 |
| pyttsx3         | 2.9           | TTS offline                        |
| eSpeak-NG       | 1.52          | Paquete del sistema (Linux) / MSI  |

*Hardware de prueba:* Intel i5-1135G7 | 8 GB RAM | Webcam 720p  
*Opcional:* Smartphone Snapdragon 720G + Chaquopy

---

## Instalación
```bash
# 1️⃣  Clona el repositorio y sitúate en la carpeta raíz
git clone <url-del-repo>
cd PROTOTIPO-TESIS

# 2️⃣  Crea y activa un entorno virtual
python -m venv env
# Linux / macOS
source env/bin/activate
# Windows
env\Scripts\activate

# 3️⃣  Instala **todas las dependencias** listadas en requirements.txt
pip install -r requirements.txt

# 4️⃣  Copia/descarga los modelos en la carpeta raíz
.
├── main.py
├── yolov8n.pt
├── yolov8n.onnx
├── midas_v21_small_256.pt
└── model-small.onnx


