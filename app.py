import os
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320MobilenetV3LargeWeights
from torchvision import transforms
from PIL import Image
import time
from datetime import datetime
import threading
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Usar puerto de Render o 5000 por defecto
PORT = int(os.environ.get('PORT', 5000))

# Configurar Cloudinary con variables de entorno
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dj7btfjld'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '965316488217193'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 'tcruSWb2pnAodSU1mGNBO5rBc9E')
)

# Contador de nuevas etiquetas (solo para simular reentrenamiento)
new_labels_count = 0
labels_threshold = 10

# Cargar modelo MobileNet SSD desde torchvision (no necesita Torch Hub)
model = None

def load_model():
    global model
    try:
        print("Cargando modelo SSDLite320MobileNetV3Large desde torchvision...")
        weights = SSDLite320MobilenetV3LargeWeights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights, progress=True)
        model.eval()
        print("Modelo SSDLite320MobileNetV3Large cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        model = None

# Intentar cargar modelo al iniciar
load_model()

# Densidades de materiales (g/cm³)
material_densities = {
    'bottle': 2.5,      # vidrio
    'cup': 1.2,         # plástico
    'banana': 0.6,      # orgánico
    'apple': 0.6,       # orgánico
    'cell phone': 2.7,  # aluminio/plástico
    'book': 1.2,        # papel
    'car': 0.8,         # metal/otros
    'chair': 0.6,       # madera/plástico
    'laptop': 2.0,      # metal/plástico
    'bowl': 1.2,        # cerámica/plástico
    'orange': 0.6,      # orgánico
    'broccoli': 0.6,    # orgánico
    'vase': 2.5,        # vidrio/cerámica
    'scissors': 2.7,    # metal
    'toothbrush': 1.2,  # plástico
    'person': 0,        # ignorar
    'car': 0.8,         # ignorar
    'bus': 0.8,         # ignorar
    'truck': 0.8,       # ignorar
    'traffic light': 0, # ignorar
    'fire hydrant': 0,  # ignorar
    'stop sign': 0,     # ignorar
    'parking meter': 0, # ignorar
    'bench': 0.6,       # madera
    'bird': 0,          # ignorar
    'cat': 0,           # ignorar
    'dog': 0,           # ignorar
    'horse': 0,         # ignorar
    'sheep': 0,         # ignorar
    'cow': 0,           # ignorar
    'elephant': 0,      # ignorar
    'bear': 0,          # ignorar
    'zebra': 0,         # ignorar
    'giraffe': 0,       # ignorar
    'backpack': 1.2,    # tela/plástico
    'umbrella': 1.2,    # tela/plástico
    'handbag': 1.2,     # cuero/plástico
    'tie': 1.2,         # tela
    'suitcase': 1.2,    # tela/plástico
    'frisbee': 1.2,     # plástico
    'skis': 0.8,        # madera/plástico
    'snowboard': 0.8,   # madera/plástico
    'sports ball': 1.2, # cuero/plástico
    'kite': 1.2,        # tela/plástico
    'baseball bat': 0.6, # madera
    'baseball glove': 1.2, # cuero
    'skateboard': 0.8,  # madera/plástico
    'surfboard': 0.8,   # espuma/plástico
    'tennis racket': 1.2, # metal/plástico
    'bottle': 2.5,      # vidrio
    'wine glass': 2.5,  # vidrio
    'cup': 1.2,         # cerámica/plástico
    'fork': 2.7,        # metal
    'knife': 2.7,       # metal
    'spoon': 2.7,       # metal
    'bowl': 2.5,        # cerámica/vidrio
    'banana': 0.6,      # orgánico
    'apple': 0.6,       # orgánico
    'sandwich': 0.6,    # orgánico
    'orange': 0.6,      # orgánico
    'broccoli': 0.6,    # orgánico
    'carrot': 0.6,      # orgánico
    'hot dog': 0.6,     # orgánico
    'pizza': 0.6,       # orgánico
    'donut': 0.6,       # orgánico
    'cake': 0.6,        # orgánico
    'chair': 0.6,       # madera/plástico
    'couch': 0.6,       # madera/tela
    'potted plant': 0.6, # cerámica/orgánico
    'bed': 0.6,         # madera/tela
    'dining table': 0.6, # madera
    'toilet': 2.3,      # cerámica
    'tv': 2.0,          # plástico/metal
    'laptop': 2.0,      # metal/plástico
    'mouse': 1.2,       # plástico
    'remote': 1.2,      # plástico
    'keyboard': 1.2,    # plástico
    'cell phone': 2.7,  # metal/plástico
    'microwave': 2.0,   # metal/plástico
    'oven': 2.0,        # metal
    'toaster': 2.0,     # metal
    'sink': 2.3,        # cerámica
    'refrigerator': 2.0, # metal/plástico
    'book': 1.2,        # papel
    'clock': 2.0,       # metal/plástico
    'vase': 2.5,        # cerámica/vidrio
    'scissors': 2.7,    # metal
    'teddy bear': 0.6,  # tela/plástico
    'hair drier': 1.2,  # plástico
    'toothbrush': 1.2,  # plástico
}

def get_material_density_and_category(object_name):
    # Mapear nombre de objeto a categoría y densidad
    if object_name in material_densities:
        density = material_densities[object_name]
        if object_name in ['bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'cell phone', 'book', 'vase', 'scissors', 'toothbrush']:
            category = 'Aprovechable'
        elif object_name in ['banana', 'apple', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'sandwich']:
            category = 'Orgánico'
        elif object_name in ['cell phone', 'battery', 'light bulb', 'paint', 'chemicals']:
            category = 'Peligroso'
        elif object_name in ['person', 'car', 'bus', 'truck', 'bench']:
            category = 'Ignorar'
        else:
            category = 'No aprovechable'
        return category, density
    else:
        return 'Desconocido', 1.0

def save_image_with_label(image, object_name):
    global new_labels_count
    
    try:
        # Convertir imagen a bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Subir imagen a Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_byte_arr,
            folder='recycling_app/learning',
            public_id=f"{object_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            resource_type='image'
        )
        
        new_labels_count += 1
        
        if new_labels_count >= labels_threshold:
            new_labels_count = 0
            threading.Thread(target=retrain_model).start()
            
        return upload_result['secure_url']
        
    except Exception as e:
        print(f"Error al procesar imagen: {e}")
        return None

def retrain_model():
    print("Iniciando reentrenamiento del modelo...")
    time.sleep(5)
    print("Modelo reentrenado exitosamente!")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if model is None:
            return jsonify({'error': 'Modelo no disponible. Contacta al administrador.'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image_data = file.read()

        # Preparar imagen para detección
        img = Image.open(BytesIO(image_data)).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)  # Añadir batch dimension

        # Hacer predicción
        with torch.no_grad():
            predictions = model(img_tensor)

        # Procesar resultados
        detected_waste = []
        for i in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][i].item()
            if score > 0.5:  # Umbral de confianza
                box = predictions[0]['boxes'][i].tolist()
                label_idx = predictions[0]['labels'][i].item()
                
                # Mapear índice de etiqueta a nombre de clase
                # Usar un diccionario de COCO
                coco_labels = [
                    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
                
                if label_idx < len(coco_labels):
                    label_name = coco_labels[label_idx]
                    
                    # Ignorar personas
                    if label_name == 'person':
                        continue
                    
                    category, density = get_material_density_and_category(label_name)
                    
                    detected_waste.append({
                        'label': label_name,
                        'confidence': score,
                        'bbox': box,
                        'category': category,
                        'density': density
                    })

        return jsonify(detected_waste)
    except Exception as e:
        print(f"Error en /detect: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/collect', methods=['POST'])
def collect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        object_name = request.form.get('object_name')

        if not object_name:
            return jsonify({'error': 'No object name provided'}), 400

        img = Image.open(BytesIO(file.read()))
        cloudinary_url = save_image_with_label(img, object_name)

        if cloudinary_url:
            return jsonify({'success': True, 'cloudinary_url': cloudinary_url})
        else:
            return jsonify({'error': 'Failed to upload image'}), 500
    except Exception as e:
        print(f"Error en /collect: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Backend de Clasificador de Residuos (con MobileNet SSD) funcionando."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)