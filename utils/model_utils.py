"""
utils/model_utils.py
---------------------
Módulo de utilidades: carga de modelos y funciones de inferencia
para el sistema de reconocimiento de serpientes.
"""

import json
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import ImageOps, Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tensorflow as tf
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ---------------------------------------------------------------------------
# Carga de Modelos
# ---------------------------------------------------------------------------
def load_presence_model(weights_path: str):
    """Carga el modelo de clasificación/detección de presencia YOLOv8 (.pt)."""
    model = YOLO(weights_path)
    return model


def load_species_model(weights_path: str, num_classes: int = 80) -> nn.Module:
    """Carga el modelo multiclase EfficientNet-B0 de especie."""
    model = models.efficientnet_b0(weights=None)
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        checkpoint.to(DEVICE).eval()
        return checkpoint

    checkpoint_num_classes = state_dict['classifier.1.weight'].shape[0] if 'classifier.1.weight' in state_dict else num_classes
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, checkpoint_num_classes)
    
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def load_venom_model(weights_path: str = "models/modelo_veneno.weights.h5"):
    """Carga el modelo Keras de veneno desde sus pesos guardados."""
    possible_paths = [
        weights_path,
        "models/modelo_veneno.weights.h5",
        os.path.join(os.path.dirname(__file__), "..", "models", "modelo_veneno.weights.h5")
    ]
    
    actual_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not actual_path:
        raise FileNotFoundError("❌ No se encontró el archivo de pesos de veneno en las rutas especificadas.")

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.load_weights(actual_path)
        return model

    except Exception as e:
        raise RuntimeError(f"Error al cargar la matriz de pesos desde {actual_path}: {e}")


def load_class_names(json_path: str = "models/class_name.json") -> list:
    """Carga los nombres de las clases desde el archivo JSON."""
    if not os.path.exists(json_path):
        return list(TRADUCCION_ESPECIES.keys())
        
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        
    return raw

# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def preprocess_for_torch(image_rgb: np.ndarray) -> torch.Tensor:
    """Aplica la transformación Albumentations y genera un tensor [1, C, H, W]."""
    augmented = _eval_transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor


def preprocess_for_keras(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocesamiento para el modelo MobileNetV2 Keras de veneno."""
    img = cv2.resize(image_rgb, (size, size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Implementación de Grad-CAM para visualizar mapas de atención."""

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        if target_layer is None:
            self.target_layer = model.features[-1]
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]
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
# Funciones de preprocesamiento y predicción ajustadas
# ---------------------------------------------------------------------------
def preprocess_high_res_image(image_rgb: np.ndarray, target_size: int = 416) -> np.ndarray:
    """
    Realiza un recorte centrado manteniendo proporción e interpola exactamente
    a la resolución objetivo de entrenamiento (416x416) usando LANCZOS.
    Evita la deformación por escala y elimina el ruido de alta resolución.
    """
    pil_img = Image.fromarray(image_rgb)
    
    # Recorte cuadrado centrado
    width, height = pil_img.size
    min_dim = min(width, height)
    
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    
    pil_img_cropped = pil_img.crop((left, top, right, bottom))
    
    # Reescalado exacto a 416x416 con resampling LANCZOS (suaviza bordes y elimina artefactos)
    pil_img_resized = pil_img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return np.array(pil_img_resized)


def predict_presence(presence_model, image_rgb, min_confidence=0.85):
    """
    Evalúa la presencia usando YOLO Classification con preprocesamiento adaptado al entrenamiento.
    """
    # 1. Ajustar a resolución exacta de entrenamiento (416x416)
    img_processed = preprocess_high_res_image(image_rgb, target_size=416)
    
    # 2. Inferencia en YOLO
    results = presence_model(img_processed, verbose=False)[0]
    
    probs = results.probs
    top1_idx = int(probs.top1)
    top1_class = str(results.names[top1_idx]).lower().strip()
    top1_conf = float(probs.top1conf.cpu())

    # 3. Lógica de decisión y calibración de probabilidad devuelta
    if top1_class == "snake" and top1_conf >= min_confidence:
        has_snake = True
        return has_snake, top1_conf
    elif top1_class == "snake" and top1_conf < min_confidence:
        # Detectó rasgos pero no supera el umbral
        has_snake = False
        return has_snake, top1_conf
    else:
        # La clase principal fue 'no_snake'
        has_snake = False
        snake_prob = 1.0 - top1_conf  # Probabilidad real asignada a que SEA serpiente
        return has_snake, snake_prob


def predict_species(model: nn.Module, image_rgb: np.ndarray, json_path: str = "models/class_name.json"):
    """Identifica la especie en español y devuelve (nombre_español, probabilidad)."""
    class_names = load_class_names(json_path)
    tensor = preprocess_for_torch(image_rgb)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    raw_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Especie_{pred_idx}"
    spanish_name = TRADUCCION_ESPECIES.get(raw_name, raw_name)
    
    return spanish_name, float(probs[pred_idx])


def predict_venom(model, image_rgb: np.ndarray, threshold: float = 0.5):
    """Clasifica si la serpiente es venenosa usando el modelo Keras."""
    x = preprocess_for_keras(image_rgb)
    raw_output = model.predict(x, verbose=0)
    prob_venomous = float(raw_output[0][0])
    is_venomous = prob_venomous >= threshold
    return is_venomous, prob_venomous


def generate_gradcam(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """Genera la imagen con el mapa de calor Grad-CAM superpuesto para la clase predicha."""
    tensor = preprocess_for_torch(image_rgb)
    grad_cam = GradCAM(model=model)
    
    with torch.enable_grad():
        logits = model(tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())
        cam_mask = grad_cam.generate(input_tensor=tensor, class_idx=pred_class)
    
    overlay_img = overlay_heatmap(image_rgb, cam_mask)
    model.zero_grad()
    
    return overlay_img
