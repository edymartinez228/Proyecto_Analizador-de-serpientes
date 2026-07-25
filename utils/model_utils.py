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


def load_venom_model(weights_path):
    # 1. Desactivar el uso de Keras 3 para evitar choques con el formato antiguo .h5
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    # Intento 1: Usar tf_keras silenciando el guardado estricto
    try:
        import tf_keras
        return tf_keras.models.load_model(weights_path, compile=False)
    except Exception:
        pass

    # Intento 2: Usar tf.keras deshabilitando compilación y safe_mode
    try:
        return tf.keras.models.load_model(weights_path, compile=False, safe_mode=False)
    except Exception:
        pass

    # Intento 3: Cargar como modelo Keras genérico deshabilitando la validación de la configuración de entrada
    try:
        from tensorflow.keras.models import load_model
        return load_model(weights_path, compile=False)
    except Exception as e:
        # Fallback final: Si falla por capas desactualizadas, cargar con custom_objects permisivo
        def custom_objects():
            return {}
        
        return tf.keras.models.load_model(
            weights_path, 
            compile=False, 
            custom_objects={"InputLayer": tf.keras.layers.InputLayer}
        )


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
