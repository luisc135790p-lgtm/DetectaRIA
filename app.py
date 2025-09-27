import os
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
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

# Cargar modelo YOLOv5 con manejo de errores y rate limit
model = None

def load_model_with_retry():
    global model
    print("Intentando cargar modelo...")
    
    # Intentar con modelo ligero primero (más probable que funcione)
    models_to_try = [
        ('yolov5n', 'modelo muy ligero'),
        ('yolov5s', 'modelo pequeño'),
        ('yolov5m', 'modelo mediano')
    ]
    
    for model_name, description in models_to_try:
        try:
            print(f"Intentando cargar {model_name} ({description})...")
            # Añadir parámetros para evitar rate limit
            model = torch.hub.load(
                'ultralytics/yolov5', 
                model_name, 
                pretrained=True, 
                trust_repo=True,
                force_reload=False  # Evitar recargas innecesarias
            )
            model.eval()
            print(f"Modelo {model_name} ({description}) cargado correctamente.")
            return True
        except Exception as e:
            print(f"Error al cargar {model_name}: {e}")
            # Esperar un poco antes de intentar el siguiente
            time.sleep(2)
    
    print("No se pudo cargar ningún modelo. La detección no funcionará.")
    return False

# Intentar cargar modelo al iniciar (con manejo de rate limit)
load_model_with_retry()

# Densidades de materiales (g/cm³)
material_densities = {
    'vidrio': 2.5,
    'aluminio': 2.7,
    'pet': 1.38,
    'papel': 1.2,
    'carton': 0.7,
    'plastico': 1.2,
    'madera': 0.6,
    'ceramica': 2.4,
}

def get_material_density(object_name):
    # Categorías por defecto
    if 'bottle' in object_name:
        return 'vidrio', 2.5
    elif 'can' in object_name:
        return 'aluminio', 2.7
    elif 'paper' in object_name or 'book' in object_name:
        return 'papel', 1.2
    else:
        return 'desconocido', 1.0

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
            return jsonify({
                'error': 'Modelo no disponible. Contacta al administrador.',
                'message': 'El modelo de detección no se pudo cargar en el servidor.'
            }), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        image_data = file.read()

        img = Image.open(BytesIO(image_data))
        
        results = model(img)
        predictions = results.pandas().xyxy[0].to_dict()

        detected_waste = []
        for i in range(len(predictions['name'])):
            name = predictions['name'][i]
            confidence = predictions['confidence'][i]
            
            if name == 'person':
                continue
                
            if confidence > 0.3:
                material, density = get_material_density(name)
                
                detected_waste.append({
                    'label': name,
                    'confidence': float(confidence),
                    'bbox': [
                        float(predictions['xmin'][i]),
                        float(predictions['ymin'][i]),
                        float(predictions['xmax'][i]),
                        float(predictions['ymax'][i])
                    ],
                    'category': material,
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
    return "Backend de Clasificador de Residuos (con Cloudinary) funcionando."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)