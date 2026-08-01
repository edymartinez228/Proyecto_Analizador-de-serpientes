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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tensorflow as tf
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
IMG_SIZE_DEFAULT = 224
IMG_SIZE_SPECIES = 260
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Palabras clave y géneros médicos de importancia para la validación cruzada de veneno
VENOMOUS_KEYWORDS = [
    # Nombres científicos (Géneros)
    "agkistrodon", "austrelaps", "bitis", "bothriechis", "bothrops", "bungarus",
    "causus", "crotalus", "daboia", "dendroaspis", "laticauda", "micrurus",
    "naja", "ophiophagus", "oxyuranus", "protobothrops", "pseudechis",
    "pseudonaja", "rhabdophis", "sistrurus", "trimeresurus", "tropidolaemus", "vipera",
    # Nombres comunes
    "cobra", "viper", "mamba", "krait", "coral", "rattlesnake", "taipan",
    "copperhead", "cottonmouth", "boomslang", "cantil", "cascabel", "víbora"
]

# Diccionario de traducción Nombre Científico -> Nombre Común en Español
TRADUCCION_ESPECIES = {
    "Agkistrodon contortrix": "Cabeza de Cobre (Copperhead)",
    "Agkistrodon piscivorus": "Boca de Algodón (Cottonmouth)",
    "Ahaetulla nasuta": "Serpiente Látigo Narizona",
    "Ahaetulla prasina": "Serpiente Látigo Verde",
    "Arizona elegans": "Serpiente Elegante de Arizona",
    "Aspidites melanocephalus": "Pitón de Cabeza Negra",
    "Atractus crassicaudatus": "Sabanera / Serpiente Tierrera",
    "Austrelaps superbus": "Serpiente Cabeza de Cobre Australiana",
    "Bitis arietans": "Víbora Sopladora (Puff Adder)",
    "Bitis gabonica": "Víbora de Gabón",
    "Boa constrictor": "Boa Constrictor / Mazacuata",
    "Bogertophis subocularis": "Serpiente Ratonera Trans-Pecos",
    "Boiga irregularis": "Serpiente Gato Marrón",
    "Boiga kraepelini": "Serpiente Gato de Kraepelin",
    "Bothriechis schlegelii": "Víbora de Pestañas / Bocaracá",
    "Bothrops asper": "Barba Amarilla / Terciopelo",
    "Bothrops atrox": "Jarca / Mapaná / Labaria",
    "Bungarus multicinctus": "Krait de Bandas Múltiples",
    "Carphophis amoenus": "Serpiente de Gusano Oriental",
    "Carphophis vermis": "Serpiente de Gusano Occidental",
    "Causus rhombeatus": "Víbora Nocturna Romboédrica",
    "Cemophora coccinea": "Serpiente Escarlata",
    "Charina bottae": "Boa de Goma del Norte",
    "Chrysopelea ornata": "Serpiente Voladora Ornamentada",
    "Clonophis kirtlandii": "Serpiente de Kirtland",
    "Contia tenuis": "Serpiente de Cola Afilada",
    "Corallus caninus": "Boa Esmeralda",
    "Corallus hortulanus": "Boa Arborícola del Amazonas",
    "Coronella girondica": "Culebra Bordelesa",
    "Crotalus adamanteus": "Cascabel Diamantina del Este",
    "Crotalus atrox": "Cascabel Diamantina del Oeste",
    "Crotalus cerastes": "Serpiente de Cascabel Cornuda (Sidewinder)",
    "Crotalus cerberus": "Cascabel Negra de Arizona",
    "Crotalus lepidus": "Cascabel de las Rocas",
    "Crotalus molossus": "Cascabel de Cola Negra",
    "Crotalus ornatus": "Cascabel Ornamental del Este",
    "Crotalus ruber": "Cascabel Roja",
    "Crotalus scutulatus": "Cascabel del Mojave",
    "Crotalus stephensi": "Cascabel de Panamaint",
    "Crotalus tigris": "Cascabel Tigre",
    "Crotalus triseriatus": "Cascabel Transvolcánica",
    "Crotalus viridis": "Cascabel Verde de las Praderas",
    "Crotaphopeltis hotamboeia": "Serpiente Heráldica / Labios Rojos",
    "Daboia russelii": "Víbora de Russell",
    "Dendrelaphis pictus": "Serpiente de Árbol Bronceada Pintada",
    "Dendrelaphis punctulatus": "Serpiente de Árbol Verde Australiana",
    "Dendroaspis polylepis": "Mamba Negra",
    "Diadophis punctatus": "Serpiente de Cuello Anillado",
    "Drymarchon couperi": "Serpiente Índigo del Este",
    "Elaphe dione": "Culebra Dione",
    "Epicrates cenchria": "Boa Arcoíris",
    "Eunectes murinus": "Anaconda Verde",
    "Farancia abacura": "Serpiente de Fango",
    "Gonyosoma oxycephalum": "Serpiente Ratón Verde de Cola Roja",
    "Hemorrhois hippocrepis": "Culebra de Herradura",
    "Heterodon nasicus": "Serpiente Hocico de Cerdo Occidental",
    "Heterodon simus": "Serpiente Hocico de Cerdo del Sureste",
    "Hierophis viridiflavus": "Culebra Verdiamarilla",
    "Hypsiglena torquata": "Serpiente Nocturna",
    "Imantodes cenchoa": "Serpiente Cordelilla / Ojo de Gato",
    "Lampropeltis alterna": "Serpiente Rey de las Rocas",
    "Lampropeltis calligaster": "Serpiente Rey Topera",
    "Lampropeltis getula": "Serpiente Rey Oriental",
    "Lampropeltis pyromelana": "Serpiente Rey de las Montañas de Sonora",
    "Lampropeltis triangulum": "Serpiente Rey Falsa Coral",
    "Lampropeltis zonata": "Serpiente Rey de California",
    "Laticauda colubrina": "Serpiente Marina de Bandas",
    "Leptodeira annulata": "Serpiente Ojo de Gato Anillada",
    "Leptophis ahaetulla": "Serpiente Lora / Bejuquilla",
    "Leptophis diplotropis": "Serpiente Lora de Cintura",
    "Leptophis mexicanus": "Serpiente Lora Mexicana",
    "Lycodon capucinus": "Serpiente Lobo Común",
    "Malpolon monspessulanus": "Culebra Bastarda",
    "Masticophis bilineatus": "Corredora de Dos Líneas",
    "Masticophis lateralis": "Corredora de Franja Verde",
    "Masticophis schotti": "Corredora de Schott",
    "Masticophis taeniatus": "Corredora Rayada",
    "Micrurus fulvius": "Coralillo Arlequín del Este",
    "Micrurus tener": "Coralillo de Texas",
    "Morelia spilota": "Pitón Alfombra",
    "Morelia viridis": "Pitón Arborícola Verde",
    "Naja atra": "Cobra de China",
    "Naja naja": "Cobra India / Cobra de Anteojos",
    "Naja nivea": "Cobra del Cabo",
    "Natrix maura": "Culebra Viperina",
    "Nerodia cyclopion": "Serpiente de Agua de Vientre Verde",
    "Nerodia floridana": "Serpiente de Agua de Florida",
    "Nerodia taxispilota": "Serpiente de Agua Diamantada",
    "Ninia sebae": "Serpiente Coral del Café",
    "Opheodrys aestivus": "Serpiente Verde Rugosa",
    "Ophiophagus hannah": "Cobra Real",
    "Oxybelis aeneus": "Serpiente Bejuquilla Marrón",
    "Oxyuranus scutellatus": "Taipán Costero",
    "Phyllorhynchus decurtatus": "Serpiente Hocico de Hoja",
    "Pituophis catenifer": "Serpiente Toro / Gopher",
    "Pituophis deppei": "Serpiente Mexicana de los Pinos",
    "Protobothrops mucrosquamatus": "Víbora de Habitáculo Habu",
    "Psammodynastes pulverulentus": "Serpiente Mock Viper / Falsa Víbora",
    "Pseudaspis cana": "Serpiente Topera Africana",
    "Pseudechis australis": "Serpiente Rey Marrón / Mulga",
    "Pseudechis porphyriacus": "Serpiente Negra de Vientre Rojo",
    "Pseudonaja textilis": "Serpiente Marrón Oriental",
    "Python molurus": "Pitón de la India",
    "Python regius": "Pitón Bola",
    "Regina septemvittata": "Serpiente Reina",
    "Rhabdophis subminiatus": "Serpiente de Cuello Rojo",
    "Rhabdophis tigrinus": "Serpiente Tigre de Agua (Yamakagashi)",
    "Rhadinaea flavilata": "Serpiente de Pino de Labios Amarillos",
    "Rhinocheilus lecontei": "Serpiente Nariz de Lenguado",
    "Salvadora grahamiae": "Serpiente Parche de Montaña",
    "Salvadora hexalepis": "Serpiente Parche del Desierto",
    "Senticolis triaspis": "Serpiente Ratonera Verde Mexicana",
    "Sistrurus catenatus": "Cascabel Massasauga",
    "Sistrurus miliarius": "Cascabel Pigmea",
    "Spilotes pullatus": "Serpiente Tigre / Toche",
    "Tantilla coronata": "Serpiente Coronado del Sureste",
    "Tantilla gracilis": "Serpiente de Cabeza Plana Esbelta",
    "Tantilla hobartsmithi": "Serpiente de Cabeza Plana del Suroeste",
    "Tantilla planiceps": "Serpiente de Cabeza Plana de California",
    "Thamnophis atratus": "Serpiente de Liga de Cuello Marrón",
    "Thamnophis couchii": "Serpiente de Liga de Sierra",
    "Thamnophis cyrtopsis": "Serpiente de Liga de Cuello Negro",
    "Thamnophis marcianus": "Serpiente de Liga Ajedrezada",
    "Thamnophis ordinoides": "Serpiente de Liga del Noroeste",
    "Thamnophis proximus": "Serpiente de Liga de Cinta",
    "Thamnophis radix": "Serpiente de Liga de las Llanuras",
    "Trimeresurus stejnegeri": "Víbora de Árbol Verde de Bamboo",
    "Tropidoclonion lineatum": "Serpiente Rayada de Rayas Amarillas",
    "Tropidolaemus subannulatus": "Víbora de Árbol del Sudeste Asiático",
    "Tropidolaemus wagleri": "Víbora del Templo de Wagler",
    "Vipera ammodytes": "Víbora cornuda europea",
    "Vipera aspis": "Víbora Áspid",
    "Vipera seoanei": "Víbora de Seoane",
    "Virginia valeriae": "Serpiente de Tierra Suave",
    "Xenochrophis piscator": "Serpiente de Agua de Cuello Rayado"
}

# ---------------------------------------------------------------------------
# Procesamiento Adaptativo de Proporciones (Letterboxing)
# ---------------------------------------------------------------------------
def resize_aspect_ratio_pad(image_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Redimensiona cualquier imagen a target_size x target_size preservando la relación de aspecto."""
    h, w = image_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
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


def load_species_model(weights_path: str, num_classes: int = None) -> nn.Module:
    """Carga el modelo EfficientNet con detección adaptativa de arquitectura."""
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

    if num_classes is None:
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
        else:
            num_classes = 135

    # Carga multiarquitectura resiliente (B0 -> B1 -> B2)
    architectures = [models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2]
    loaded_model = None

    for arch_func in architectures:
        try:
            model = arch_func(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            model.load_state_dict(state_dict)
            loaded_model = model
            break
        except RuntimeError:
            continue

    if loaded_model is None:
        raise RuntimeError("❌ No se pudo hacer encajar el archivo .pth con ninguna versión de EfficientNet (B0, B1, B2).")

    loaded_model.to(DEVICE).eval()
    return loaded_model


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


def load_class_names(json_path: str = "modelo_especie_out/class_names.json") -> list:
    """Carga los nombres de las clases desde el JSON."""
    possible_paths = [
        json_path,
        "modelo_especie_out/class_names.json",
        "models/class_names.json",
        "models/class_name.json"
    ]
    actual_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not actual_path:
        return list(TRADUCCION_ESPECIES.keys())
        
    with open(actual_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
        
    return raw

# ---------------------------------------------------------------------------
# Preprocesamiento Adaptativo
# ---------------------------------------------------------------------------
def preprocess_for_torch(image_rgb: np.ndarray, target_size: int = IMG_SIZE_SPECIES) -> torch.Tensor:
    """Adapta cualquier resolución al target_size deseado respetando la relación de aspecto."""
    padded_img = resize_aspect_ratio_pad(image_rgb, target_size=target_size)
    
    eval_transform = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    augmented = eval_transform(image=padded_img)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor


def preprocess_for_keras(image_rgb: np.ndarray, target_size: int = IMG_SIZE_DEFAULT) -> np.ndarray:
    """Adapta cualquier resolución al tamaño Keras preservando aspecto y normalizando [0,1]."""
    padded_img = resize_aspect_ratio_pad(image_rgb, target_size=target_size)
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
    """Superpone el mapa de calor sobre la resolución original de la imagen."""
    orig_h, orig_w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(original_rgb * (1 - alpha) + heatmap * alpha)
    return overlay

# ---------------------------------------------------------------------------
# Funciones de Predicción e Inferencia Integradas
# ---------------------------------------------------------------------------
def predict_species(model: nn.Module, image_rgb: np.ndarray, json_path: str = "modelo_especie_out/class_names.json", top_k: int = 5):
    """Identifica la especie y devuelve el ranking Top-K (por defecto Top 5) de predicciones traducidas."""
    class_names = load_class_names(json_path)
    tensor = preprocess_for_torch(image_rgb, target_size=IMG_SIZE_SPECIES)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_k_count = min(top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k_count]
    
    predictions = []
    for idx in top_indices:
        raw_name = class_names[idx] if idx < len(class_names) else f"Especie_{idx}"
        spanish_name = TRADUCCION_ESPECIES.get(raw_name, raw_name)
        predictions.append({
            "raw_name": raw_name,
            "spanish_name": spanish_name,
            "probability": float(probs[idx])
        })

    top_pred = predictions[0]
    return top_pred["spanish_name"], top_pred["probability"], predictions


def predict_species(model: nn.Module, image_rgb: np.ndarray, json_path: str = "modelo_especie_out/class_names.json", top_k: int = 3):
    """Identifica la especie y devuelve el ranking Top-K de predicciones traducidas."""
    class_names = load_class_names(json_path)
    tensor = preprocess_for_torch(image_rgb, target_size=IMG_SIZE_SPECIES)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_k_count = min(top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k_count]
    
    predictions = []
    for idx in top_indices:
        raw_name = class_names[idx] if idx < len(class_names) else f"Especie_{idx}"
        spanish_name = TRADUCCION_ESPECIES.get(raw_name, raw_name)
        predictions.append({
            "raw_name": raw_name,
            "spanish_name": spanish_name,
            "probability": float(probs[idx])
        })

    top_pred = predictions[0]
    return top_pred["spanish_name"], top_pred["probability"], predictions


def predict_venom(model, image_rgb: np.ndarray, threshold: float = 0.5):
    """Clasifica veneno en 224x224 manteniendo la relación de aspecto."""
    x = preprocess_for_keras(image_rgb, target_size=IMG_SIZE_DEFAULT)
    raw_output = model.predict(x, verbose=0)
    prob_venomous = float(raw_output[0][0])
    is_venomous = prob_venomous >= threshold
    return is_venomous, prob_venomous


def cross_validate_venom_risk(species_raw_name: str, is_venomous_pred: bool) -> dict:
    """Validación Cruzada de Seguridad (Especie vs Veneno)."""
    species_lower = species_raw_name.lower()
    is_known_venomous_species = any(kw in species_lower for kw in VENOMOUS_KEYWORDS)
    
    warning_triggered = False
    warning_message = ""

    if is_known_venomous_species and not is_venomous_pred:
        warning_triggered = True
        warning_message = (
            f"⚠️ ADVERTENCIA DE SEGURIDAD: La especie identificada '{species_raw_name}' "
            "pertenece a un género o grupo de serpientes potencialmente VENENOSAS, "
            "aunque el clasificador de veneno indicó un riesgo bajo. "
            "Mantenga la distancia y tome precauciones extremas."
        )

    return {
        "warning_triggered": warning_triggered,
        "warning_message": warning_message
    }


def generate_gradcam(model: nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """Genera la visualización Grad-CAM proyectada en la dimensión original."""
    tensor = preprocess_for_torch(image_rgb, target_size=IMG_SIZE_SPECIES)
    grad_cam = GradCAM(model=model)
    
    with torch.enable_grad():
        logits = model(tensor)
        pred_class = int(torch.argmax(logits, dim=1).item())
        cam_mask = grad_cam.generate(input_tensor=tensor, class_idx=pred_class)
    
    overlay_img = overlay_heatmap(image_rgb, cam_mask)
    model.zero_grad()
    
    return overlay_img
