"""
utils/model_utils.py
---------------------
Módulo de utilidades: Carga de modelos y funciones de inferencia
con soporte adaptativo de resolución (Letterboxing) para prevenir
deformaciones y falsos positivos.
"""

import json
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image, ImageOps
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

# Diccionario de traducción inglés -> español para las clases del modelo
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
# Procesamiento Adaptativo de Proporciones (Letterboxing)
# ---------------------------------------------------------------------------
def resize_aspect_ratio_pad(image_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Redimensiona cualquier imagen (miniaturas, panorámicas o 4K) a target_size x target_size
    preservando la relación de aspecto exacta mediante bordes neutros (Padding).
    Evita la distorsión anatómica del objeto.
    """
    h, w = image_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Selección del algoritmo de interpolación según subida o bajada de escala
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    # Crear lienzo cuadrado con relleno gris neutro (128, 128, 128)
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    
    # Centrar la imagen dentro del lienzo
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas

# ---------------------------------------------------------------------------
# Carga de Modelos
# ---------------------------------------------------------------------------
def load_presence_model(weights_path: str):
    """Carga el modelo de detección de presencia YOLOv8 (.pt)."""
    model = YOLO(weights_path)
    return model


def load_species_model(weights_path: str, num_classes: int = 80) -> nn.Module:
    """Carga el modelo EfficientNet-B0 de especie."""
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
    """Carga los pesos del modelo Keras de veneno."""
    possible_paths = [
        weights_path,
        "models/modelo_veneno.weights.h5",
        os.path.join(os.path.dirname(__file__), "..", "models", "modelo_veneno.weights.h5")
    ]
    
    actual_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not actual_path:
        raise FileNotFoundError("❌ No se encontró el archivo de pesos de veneno.")

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
# Preprocesamiento Adaptativo
# ---------------------------------------------------------------------------
def preprocess_for_torch(image_rgb: np.ndarray) -> torch.Tensor:
    """Adapta cualquier resolución a 224x224 preservando aspecto y normalizando PyTorch."""
    padded_img = resize_aspect_ratio_pad(image_rgb, target_size=IMG_SIZE)
    
    eval_transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    augmented = eval_transform(image=padded_img)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor


def preprocess_for_keras(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Adapta cualquier resolución a 224x224 preservando aspecto y normalizando [0,1] Keras."""
    padded_img = resize_aspect_ratio_pad(image_rgb, target_size=size)
    img = padded_img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------------------------------------------------------------------------
# Grad-CAM Adaptativo
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
    """Superpone el mapa de calor proyectando sobre la resolución original de la imagen."""
    orig_h, orig_w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(original_rgb * (1 - alpha) + heatmap * alpha)
    return overlay

# ---------------------------------------------------------------------------
# Funciones de Predicción Adaptativas
# ---------------------------------------------------------------------------
def predict_presence(presence_model, image_rgb: np.ndarray, min_confidence: float = 0.60):
    """
    Evalúa si hay una serpiente utilizando la salida exacta del modelo YOLOv8-cls
    entrenado con las clases [0: 'no_snake', 1: 'snake'].
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("La imagen de entrada está vacía o no es válida.")

    # 1. Ajuste de dimensiones respetando el letterbox
    target_size = int(presence_model.overrides.get("imgsz", 416))
    padded_img = resize_aspect_ratio_pad(image_rgb, target_size=target_size)
    pil_image = Image.fromarray(padded_img)

    # 2. Inferencia con YOLO
    results = presence_model(pil_image, imgsz=target_size, verbose=False)[0]
    
    # Obtener el mapa de nombres ({0: 'no_snake', 1: 'snake'})
    names_map = results.names
    
    # Encontrar el índice dinámicamente según el nombre de la clase
    snake_idx = None
    for idx, class_name in names_map.items():
        if str(class_name).lower().strip() == "snake":
            snake_idx = int(idx)
            break
            
    # Si por algún motivo no la encuentra por nombre, asumimos el índice 1
    if snake_idx is None:
        snake_idx = 1

    # 3. Extraer la probabilidad pura de que sea una serpiente
    all_probs = results.probs.data.cpu().numpy()
    snake_probability = float(all_probs[snake_idx])

    # 4. Determinar si supera el umbral configurado
    has_snake = snake_probability >= min_confidence

    return has_snake, snake_probability


def predict_species(model: nn.Module, image_rgb: np.ndarray, json_path: str = "models/class_name.json"):
    """Identifica la especie adaptando resoluciones mediante padding."""
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
    """Clasifica veneno manteniendo la relación de aspecto en cualquier resolución."""
    x = preprocess_for_keras(image_rgb)
    raw_output = model.predict(x, verbose=0)
    prob_venomous = float(raw_output[0][0])
    is_venomous = prob_venomous >= threshold
    return is_venomous, prob_venomous


def generate_gradcam(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """Genera la visualización Grad-CAM proyectada exactamente en la dimensión original."""
    tensor = preprocess_for_torch(image_rgb)
    grad_cam = GradCAM(model=model)
    
    with torch.enable_grad():
        logits = model(tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())
        cam_mask = grad_cam.generate(input_tensor=tensor, class_idx=pred_class)
    
    overlay_img = overlay_heatmap(image_rgb, cam_mask)
    model.zero_grad()
    
    return overlay_img
