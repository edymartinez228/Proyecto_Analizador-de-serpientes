"""
utils/model_utils.py
---------------------
Módulo de utilidades: carga de modelos y funciones de inferencia
para el sistema de reconocimiento de serpientes.

Contiene:
    - Carga de los 2 modelos PyTorch (presencia y especie).
    - Carga del modelo Keras/TensorFlow (veneno).
    - Preprocesamiento de imágenes coherente con el entrenamiento.
    - Implementación de Grad-CAM para explicabilidad.
    - Funciones de predicción de alto nivel usadas por app.py.
"""

import json
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debe coincidir EXACTAMENTE con la transformación de evaluación usada
# durante el entrenamiento (ver notebook de entrenamiento, Paso 3).
_eval_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)

# Diccionario de traducción inglés -> español para las 80 clases del modelo
TRADUCCION_ESPECIES = {
    "Abaco island boa": "Boa de la Isla Ábaco",
    "Amazon tree boa": "Boa arborícola del Amazonas",
    "Andaman cat snake": "Serpiente gato de Andamán",
    "Andaman cobra": "Cobra de Andamán",
    "Arabian cobra": "Cobra arábiga",
    "Arizona Coral snake": "Coral de Arizona",
    "Asian cobra": "Cobra asiática",
    "Australian tiger snake": "Serpiente tigre australiana",
    "Ball python": "Pitón bola",
    "Banded Krait": "Krait bandeado",
    "Banded cat eyed snake": "Serpiente ojo de gato bandeada",
    "Banded water cobra": "Cobra de agua bandeada",
    "Beddome cat snake": "Serpiente gato de Beddome",
    "Black headed Python": "Pitón de cabeza negra",
    "Black necked spitting cobra": "Cobra escupidora de cuello negro",
    "Black racer snake": "Corredora negra",
    "Black rat snake": "Serpiente ratonera negra",
    "Black snake": "Serpiente negra",
    "Black tree cobra": "Cobra arborícola negra",
    "Boa constrictor": "Boa constrictor / Mazacuata",
    "Boiga": "Boiga / Serpiente gato",
    "Boomslang": "Boomslang / Serpiente del árbol",
    "Brahminy blind snake": "Serpiente ciega de Brahminy",
    "Brazilian coral snake": "Coral brasilera",
    "Bull snake": "Serpiente toro",
    "Canebrake": "Cascabel de los cañaverales (Canebrake)",
    "Cantil": "Cantil / Mokasin",
    "Cape cobra": "Cobra del Cabo",
    "Caspian cobra": "Cobra del Caspio",
    "Collett snake": "Serpiente de Collett",
    "Dekay Brown snake": "Serpiente marrón de DeKay",
    "Dumeril Blackheaded snake": "Serpiente de cabeza negra de Duméril",
    "Eastern Brown Snake": "Serpiente marrón oriental",
    "Emerald boa": "Boa esmeralda",
    "Equatorial spitting cobra": "Cobra escupidora ecuatorial",
    "Eqyptian cobra": "Cobra egipcia",
    "Eyelash viper": "Víbora de pestañas / Culebra de pestaña",
    "False cobra": "Falsa cobra",
    "False coral snake": "Falsa coral",
    "Fierce snake": "Taipán del interior / Serpiente feroz",
    "Forest cobra": "Cobra del bosque",
    "Forsten cat snake": "Serpiente gato de Forsten",
    "Gold ringed cat snake": "Serpiente gato de anillos dorados",
    "Green cat eyed snake": "Serpiente ojo de gato verde",
    "Grey cat snake": "Serpiente gato gris",
    "Harlequin coral snake": "Coral arlequín",
    "Hog island boa": "Boa de Hog Island",
    "Indian cobra": "Cobra india / Cobra de anteojos",
    "Indian egg eater": "Comedora de huevos india",
    "Jamaican boa": "Boa de Jamaica",
    "Javan spitting cobra": "Cobra escupidora de Java",
    "King cobra": "Cobra real",
    "Madagascar tree boa": "Boa arborícola de Madagascar",
    "Malayan blue coral snake": "Coral azul de Malasia",
    "Monocled cobra": "Cobra de monóculo",
    "Mozambique cobra": "Cobra de Mozambique",
    "Nicobar cat snake": "Serpiente gato de Nicobar",
    "Puerto rican boa": "Boa de Puerto Rico",
    "Rainbow boa": "Boa arcoíris",
    "Red spitting cobra": "Cobra escupidora roja",
    "Red tailed boa": "Boa de cola roja",
    "Red-bellied black snake": "Serpiente negra de vientre rojo",
    "Rosy boa": "Boa rosada",
    "Rubber boa": "Boa de goma",
    "Rufuos beaked snake": "Serpiente picuda rufa",
    "Sand boa": "Boa de arena",
    "Sir lanka cat snake": "Serpiente gato de Sri Lanka",
    "Snouted cobra": "Cobra hocicuda",
    "Spectacled cobra": "Cobra de anteojos",
    "Spitting cobra": "Cobra escupidora",
    "Tawny cat snake": "Serpiente gato leonada",
    "Texas blind snake": "Serpiente ciega de Texas",
    "Texas coral snake": "Coral de Texas",
    "Western blind snake": "Serpiente ciega occidental",
    "Yellow cobra": "Cobra amarilla",
    "Zebra spitting cobra": "Cobra escupidora cebra",
    "copperhead": "Cabeza de cobre (Copperhead)",
    "nubian spitting cobra": "Cobra escupidora de Nubia",
    "ornate flying snake": "Serpiente voladora ornamentada",
    "red sand boa": "Boa de arena roja"
}

def predict_species(species_model, image, json_path="models/class_name.json"):
    # 1. Cargar las clases originales del JSON
    with open(json_path, 'r') as f:
        class_names = json.load(f)

    # 2. Preprocesar la imagen (ajusta dimensiones según tu entrenamiento, ej: 224x224)
    image_resized = tf.image.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized / 255.0, axis=0)

    # 3. Predecir
    predictions = species_model.predict(image_array)
    idx = np.argmax(predictions[0])
    prob = predictions[0][idx]

    # 4. Obtener nombre en inglés
    raw_name = class_names[idx]

    # 5. Traducir al español (si no está en el mapa, muestra el nombre original)
    spanish_name = TRADUCCION_ESPECIES.get(raw_name, raw_name)

    return spanish_name, float(prob)

# ---------------------------------------------------------------------------
# Construcción y carga de modelos PyTorch (EfficientNet-B0)
# ---------------------------------------------------------------------------
def _build_efficientnet(num_classes: int) -> nn.Module:
    """Recrea la arquitectura exacta usada en el entrenamiento (sin pesos preentrenados,
    ya que se cargará el checkpoint propio a continuación)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_presence_model(weights_path: str) -> nn.Module:
    """Carga el modelo binario de presencia (Snake vs Non-Snake)."""
    model = _build_efficientnet(num_classes=2)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_species_model(weights_path, num_classes=80):
    # 1. Instanciar EfficientNet-B0
    model = models.efficientnet_b0(weights=None)
    
    # 2. Cargar los pesos guardados
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # 3. Detectar dinámicamente la cantidad de clases en el checkpoint (80)
    checkpoint_num_classes = state_dict['classifier.1.weight'].shape[0]
    
    # 4. Ajustar la capa lineal a las clases del checkpoint
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, checkpoint_num_classes)
    
    # 5. Cargar el state_dict sin errores
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    return model


def load_venom_model(weights_path="models/modelo_veneno.weights.h5"):
    possible_paths = [
        weights_path,
        "models/modelo_veneno.weights.h5",
        os.path.join(os.path.dirname(__file__), "..", "models", "modelo_veneno.weights.h5")
    ]
    
    actual_path = None
    for p in possible_paths:
        if os.path.exists(p):
            actual_path = p
            break

    if not actual_path:
        raise FileNotFoundError(f"❌ No se encontró el archivo de pesos en ninguna de las rutas intentadas.")

    try:
        # Reconstrucción de la arquitectura exacta de tu Colab (Sequential)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),  # O el valor de dropout que usaste (ej: 0.2, 0.3)
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Cargar los pesos guardados en el modelo reconstruido
        model.load_weights(actual_path)
        return model

    except Exception as e:
        raise RuntimeError(f"Error al cargar las matrices de pesos desde {actual_path}: {e}")


def load_class_names(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Si el JSON es una lista ["Abaco...", "Amazon..."]
    if isinstance(raw, list):
        return raw
    
    # Si el JSON es un diccionario {"0": "Abaco...", "1": "Amazon..."}
    if isinstance(raw, dict):
        return [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        
    return raw

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def preprocess_for_torch(image_rgb: np.ndarray) -> torch.Tensor:
    """Aplica el mismo preprocesamiento usado en entrenamiento y devuelve un tensor [1, C, H, W]."""
    augmented = _eval_transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor


def preprocess_for_keras(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocesamiento para el modelo Keras de veneno.

    ⚠️ IMPORTANTE: ajusta este preprocesamiento (tamaño, escala, normalización)
    para que coincida EXACTAMENTE con el usado al entrenar `venomous_snake_model.h5`.
    Aquí se asume resize a 224x224 y escalado a [0, 1].
    """
    img = cv2.resize(image_rgb, (size, size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM genérico mediante hooks forward/backward sobre una capa convolucional objetivo."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone el mapa de calor de Grad-CAM sobre la imagen original."""
    cam_resized = cv2.resize(cam, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(original_rgb * (1 - alpha) + heatmap * alpha)
    return overlay


# ---------------------------------------------------------------------------
# Funciones de predicción de alto nivel
# ---------------------------------------------------------------------------
def predict_presence(model: nn.Module, tensor: torch.Tensor, threshold: float = 0.6):
    """Devuelve (es_serpiente: bool, confianza: float)."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        snake_confidence = probs[1].item()  # índice 1 = clase "Snake"
    return snake_confidence >= threshold, snake_confidence


def predict_species(model: nn.Module, tensor: torch.Tensor, class_names: dict):
    """Devuelve (especie_predicha, confianza, top3 [(nombre, prob), ...], indice_predicho)."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [(class_names[int(i)], float(probs[i])) for i in top3_idx]
    return class_names[pred_idx], float(probs[pred_idx]), top3, pred_idx


def predict_venom(model, image_rgb: np.ndarray, threshold: float = 0.5):
    """Devuelve (es_venenosa: bool, confianza: float).

    Asume salida sigmoide de 1 neurona (probabilidad de "venenosa").
    Si tu modelo usa softmax de 2 neuronas [no_venenosa, venenosa], reemplaza por:
        probs = model.predict(x, verbose=0)[0]
        is_venomous = probs[1] >= threshold
        confidence = probs[1] if is_venomous else probs[0]
    """
    x = preprocess_for_keras(image_rgb)
    raw_output = model.predict(x, verbose=0)
    prob_venomous = float(raw_output[0][0])
    is_venomous = prob_venomous >= threshold
    confidence = prob_venomous if is_venomous else (1 - prob_venomous)
    return is_venomous, confidence
