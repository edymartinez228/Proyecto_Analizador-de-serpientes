# Snakely AI · Analizador de Serpientes

Aplicación web construida con Streamlit que identifica la especie de una serpiente a partir de una fotografía y estima si presenta rasgos morfológicos compatibles con toxicidad. El sistema no depende de un único modelo: cruza el resultado de dos redes neuronales entrenadas de forma independiente y resuelve cualquier discrepancia entre ambas siempre a favor del escenario más seguro para el usuario.


## Índice

1. [Descripción general](#descripción-general)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Requisitos previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Ejecución](#ejecución)
7. [Uso de la aplicación](#uso-de-la-aplicación)
8. [Entrenamiento de los modelos](#entrenamiento-de-los-modelos)
9. [Notas técnicas y limitaciones conocidas](#notas-técnicas-y-limitaciones-conocidas)
10. [Solución de problemas](#solución-de-problemas)

---

## Descripción general

Snakely AI recibe una fotografía de un ejemplar y ejecuta un pipeline de inferencia compuesto por dos modelos independientes:

- **Clasificador taxonómico** (PyTorch, familia EfficientNet): identifica la especie entre 135 clases y devuelve las 5 candidatas más probables con su nivel de confianza.
- **Detector de toxicidad** (TensorFlow/Keras, MobileNetV2): estima, sobre la propia imagen, la probabilidad de que el ejemplar presente rasgos visuales asociados a especies venenosas.

Ambos dictámenes se contrastan mediante una capa de validación cruzada (`cross_validate_venom_risk` en `utils/model_utils.py`) que compara la especie detectada contra un listado de géneros venenosos/no venenosos conocidos. Si un modelo dice "no venenosa" y el otro "venenosa", el sistema no promedia ni descarta: **eleva el nivel de alerta**, bajo el criterio de que ante una posible mordedura es preferible una falsa alarma a un falso negativo.

La interfaz también expone:

- Un control de sensibilidad para ajustar el umbral de decisión del detector de veneno.
- Mapas de atención (Grad-CAM) sobre el clasificador de especie, para visualizar qué región de la imagen influyó en la predicción.
- Un protocolo de primeros auxilios (qué hacer / qué no hacer) cuando el dictamen final es de riesgo.
- Generación de un informe descargable en HTML con el resumen del análisis.

## Arquitectura del sistema

```
                     ┌───────────────────────┐
   Imagen (JPG/PNG)  │        app.py         │
   ───────────────▶  │   (interfaz Streamlit)│
                     └──────────┬────────────┘
                                │
                 ┌──────────────┴───────────────┐
                 ▼                               ▼
     ┌───────────────────────┐      ┌────────────────────────┐
     │  Clasificador de      │      │  Detector de toxicidad │
     │  especie (PyTorch)    │      │  (TensorFlow/Keras)    │
     │  EfficientNet-B0/1/2  │      │  MobileNetV2 + MLP     │
     │  modelo_especie.pth   │      │  modelo_veneno.weights │
     └───────────┬───────────┘      └────────────┬────────────┘
                 │  top-5 especies                │ prob. de veneno
                 └───────────────┬─────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  Validación cruzada de riesgo  │
                 │  (utils/model_utils.py)        │
                 └───────────────┬─────────────────┘
                                 ▼
                 Dictamen final + Grad-CAM + informe
```

Puntos de diseño relevantes:

- **Letterboxing en el preprocesamiento**: en lugar de deformar la imagen al redimensionarla, `resize_aspect_ratio_pad` conserva la relación de aspecto y rellena el sobrante con un borde neutro. Esto evita que serpientes fotografiadas en formatos muy alargados (algo común en este tipo de fauna) generen predicciones erráticas por distorsión.
- **Detección adaptativa de arquitectura**: al cargar el `.pth` del clasificador de especie, `load_species_model` prueba EfficientNet-B0, B1 y B2 hasta encontrar la que encaja con el `state_dict` guardado, en vez de asumir una versión fija.
- **Caché de modelos**: ambos modelos se cargan una sola vez por sesión mediante `st.cache_resource`, para no reconstruir la red en cada imagen subida.
- **`app.py` es autosuficiente**: genera su propio `.streamlit/config.toml` en el primer arranque si no existe, así que no hace falta versionar ese archivo de configuración por separado.

## Estructura del proyecto

```
Proyecto_Analizador-de-serpientes-main/
├── app.py                                        # Interfaz Streamlit + orquestación del pipeline
├── requirements.txt                               # Dependencias del proyecto
├── README.md
├── .gitattributes                                 # Reglas de Git LFS para los pesos .h5
├── assets/
│   └── class_names.json                           # Las 135 clases del clasificador de especie
├── models/
│   ├── modelo_especie.pth                         # Pesos del clasificador de especie (PyTorch)
│   └── modelo_veneno.weights.h5                    # Pesos del detector de toxicidad (Keras)
├── utils/
│   └── model_utils.py                             # Carga de modelos, inferencia, Grad-CAM, validación cruzada
├── Entrenamiento_de_especies_de_serpiente.ipynb    # Notebook de entrenamiento del clasificador de especie
└── Entrenamiento_serpientes_venenosas.ipynb        # Notebook de entrenamiento del detector de toxicidad
```

Este proyecto no utiliza una base de datos: no hay datos transaccionales que persistir, el "estado" de la aplicación son los pesos entrenados de los modelos. El equivalente a los scripts de creación de base de datos son los dos notebooks de entrenamiento, que documentan de punta a punta cómo se generaron `modelo_especie.pth` y `modelo_veneno.weights.h5` a partir de los datasets públicos.

## Requisitos previos

- Python 3.10 u 3.11 (el proyecto se probó con TensorFlow 2.x y PyTorch 2.3, que aún no tienen soporte estable en 3.12+ en todas las plataformas).
- pip actualizado.
- Git con soporte para **Git LFS** (ver sección de [notas técnicas](#notas-técnicas-y-limitaciones-conocidas), es importante).
- Al menos 2 GB de RAM libres para cargar ambos modelos en memoria.
- No se requiere GPU: la app corre en CPU (se usa GPU automáticamente si `torch.cuda.is_available()` la detecta).

## Instalación

1. Clona o descomprime el proyecto y entra a la carpeta:

   ```bash
   cd Proyecto_Analizador-de-serpientes-main
   ```

2. Crea un entorno virtual (recomendado, para no ensuciar el intérprete global):

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` incluye:

   | Paquete | Uso en el proyecto |
   |---|---|
   | `streamlit` | Interfaz web y estado de sesión |
   | `torch`, `torchvision` | Clasificador de especie (EfficientNet) e inferencia Grad-CAM |
   | `tensorflow` | Detector de toxicidad (MobileNetV2) |
   | `opencv-python-headless` | Redimensionado con letterboxing y overlay del heatmap |
   | `Pillow` | Carga y corrección EXIF de la imagen subida |
   | `numpy`, `pandas` | Manejo de arreglos y datos tabulares |
   | `ultralytics`, `onnxruntime`, `onnxslim` | Dependencias de soporte usadas durante la etapa de experimentación con detección |

## Ejecución

```bash
streamlit run app.py
```

Streamlit levantará un servidor local (por defecto en `http://localhost:8501`) y abrirá la aplicación en el navegador. La primera carga tarda algunos segundos más de lo normal porque es cuando se instancian y cachean ambos modelos.

## Uso de la aplicación

1. Sube una fotografía en formato JPG, PNG o JPEG del ejemplar (idealmente con el cuerpo completo visible, buena iluminación y una sola serpiente por imagen).
2. Ajusta, si lo deseas, el **umbral de decisión de toxicidad**: a partir de qué porcentaje de indicios el sistema marca el resultado como "peligrosa". Un umbral bajo es más conservador (avisa antes, a costa de más falsas alarmas); uno alto es más estricto.
3. Activa o desactiva la generación del mapa de atención (Grad-CAM) según si te interesa ver en qué zona de la imagen se fijó el modelo.
4. Revisa el dictamen: índice de toxicidad, especie predominante, nivel de consenso entre ambos modelos y, si corresponde, el protocolo de primeros auxilios.
5. Descarga el informe en HTML desde el botón al final de la página si necesitas conservar el resultado.

## Entrenamiento de los modelos

Los pesos incluidos en `models/` ya están entrenados y listos para inferencia; no es necesario reentrenar nada para ejecutar la aplicación. Los notebooks se incluyen como evidencia y documentación del proceso de entrenamiento:

- **`Entrenamiento_de_especies_de_serpiente.ipynb`**: entrena el clasificador de 135 especies (EfficientNet-B0/B1/B2 sobre PyTorch) usando el dataset público [`goelyash/165-different-snakes-species`](https://www.kaggle.com/datasets/goelyash/165-different-snakes-species) de Kaggle.
- **`Entrenamiento_serpientes_venenosas.ipynb`**: entrena el detector binario de toxicidad (transfer learning sobre MobileNetV2, TensorFlow/Keras) usando el dataset [`adityasharma01/snake-dataset-india`](https://www.kaggle.com/datasets/adityasharma01/snake-dataset-india) de Kaggle.

Para reentrenar cualquiera de los dos, se necesita una cuenta de Kaggle con su API token configurado (`kaggle.json` o variable de entorno `KAGGLE_API_TOKEN`) y ejecutar el notebook correspondiente en un entorno con GPU (Google Colab, por ejemplo, es donde fueron desarrollados originalmente).

## Notas técnicas y limitaciones conocidas

- **Git LFS en `modelo_veneno.weights.h5`**: este archivo está versionado con Git LFS (ver `.gitattributes`). Si el proyecto se descarga como ZIP directamente desde GitHub sin resolver los punteros LFS, ese archivo llega como un texto plano de ~130 bytes en lugar del binario real (~10.3 MB), y la aplicación no podrá cargar el modelo. Antes de ejecutar, verifica el tamaño del archivo:

  ```bash
  ls -lh models/modelo_veneno.weights.h5
  ```

  Si pesa unos pocos cientos de bytes en vez de varios megabytes, resuélvelo con:

  ```bash
  git lfs install
  git lfs pull
  ```

  o descarga el binario manualmente desde el repositorio y reemplaza el archivo en `models/`.

- **Umbral de decisión ajustable, no reentrenable**: el slider de sensibilidad cambia el punto de corte sobre la salida del modelo (0.30–0.70), no reentrena ni recalibra la red.
- **Validación cruzada basada en reglas**: el cruce entre especie y toxicidad usa listas de géneros conocidos (`VENOMOUS_KEYWORDS` / `NON_VENOMOUS_KEYWORDS`), no un tercer modelo. Es una capa de seguridad determinística sobre los dos modelos, no infalible ante especies fuera de esas listas.
- **Sin persistencia entre sesiones**: `st.session_state["runs"]` solo cuenta análisis dentro de la sesión activa del navegador; no se guarda historial en disco ni en base de datos.

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `FileNotFoundError: No se encontró 'models/modelo_veneno.weights.h5'` | El archivo no existe o el directorio de trabajo no es la raíz del proyecto | Ejecuta `streamlit run app.py` desde la carpeta raíz; confirma que `models/` tiene ambos archivos |
| Error al cargar los pesos de veneno / el modelo carga pero da resultados sin sentido | El `.h5` es en realidad un puntero de Git LFS sin resolver | Ver sección anterior, `git lfs pull` |
| `RuntimeError` al hacer `load_state_dict` en el clasificador de especie | El `.pth` no corresponde a ninguna variante EfficientNet B0/B1/B2 | Confirma que `modelo_especie.pth` es el archivo original entregado, sin recortar ni sobrescribir |
| La app tarda mucho en el primer análisis | Carga inicial de TensorFlow y PyTorch en frío | Comportamiento esperado; las siguientes ejecuciones usan el modelo cacheado |
| `ModuleNotFoundError` al iniciar | Entorno virtual no activado o dependencias no instaladas | Repite el paso 3 de [Instalación](#instalación) dentro del entorno virtual activo |
