# ComfyUI UVR5 Custom Nodes

Este repositorio contiene un conjunto de nodos personalizados (Custom Nodes) para **ComfyUI** que permiten integrar **El Extractor: UVR5 (Ultimate Vocal Remover)** en tus flujos de trabajo. Estos nodos facilitan la extracción de audio desde archivos de audio o video (de forma individual o por lotes) y la separación de voces utilizando el modelo MDX-Net, específicamente el modelo `Kim_Vocal_2`.

---

## Características

* **Soporte Universal de Medios**: Compatible tanto con formatos de audio estándar (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.wma`) como de video (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`).
* **Extracción Automática de Video**: Si cargas un video, el nodo se encarga de extraer y normalizar la pista de audio internamente sin necesidad de usar herramientas externas.
* **Procesamiento en Lote**: Posibilidad de escanear un directorio entero y procesar archivos iterativamente.
* **Separación de Alta Calidad**: Integración de la librería `audio-separator` optimizada para usar la GPU, devolviendo un audio limpio con la voz separada del instrumental.
* **Compatibilidad con ComfyUI Manager**: Los nodos están configurados para ser detectados y cargados fácilmente por el Manager.

---

## Requisitos y Dependencias

Para que el extractor funcione a máxima velocidad en tu GPU, se requiere tener PyTorch con soporte CUDA ya instalado en tu entorno de ComfyUI. Además, estos nodos requieren de algunas dependencias adicionales de Python.

Las dependencias clave están en el archivo `requirements.txt`:
- `audio-separator[gpu]`
- `moviepy`
- `soundfile`
- `numpy`

### Instalación de ffmpeg (Requerido para MoviePy)
Dado que utilizamos `moviepy` para la manipulación y extracción de video, asegúrate de tener `ffmpeg` instalado en tu sistema:
- **Windows**: [Descargar ffmpeg](https://ffmpeg.org/download.html) o instalar usando scoop (`scoop install ffmpeg`) o winget (`winget install ffmpeg`).
- **Linux**: `sudo apt install ffmpeg`
- **Mac**: `brew install ffmpeg`

---

## Instalación

### Método 1: ComfyUI Manager (Recomendado)
*Próximamente disponible a través del sistema de búsqueda e instalación del Manager.*

### Método 2: Instalación Manual
1. Abre tu terminal o línea de comandos.
2. Navega hasta el directorio de nodos personalizados de tu instalación de ComfyUI:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
3. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ComfyUI-UVR5
   ```
4. Instala las dependencias requeridas en el entorno virtual de Python que usa tu ComfyUI. Si usas un entorno virtual estándar o Conda:
   ```bash
   pip install -r requirements.txt
   ```
   *(Nota para usuarios de ComfyUI portable en Windows: debes usar el ejecutable `python_embeded\python.exe` incluido en tu instalación para correr el comando de pip).*

5. Reinicia ComfyUI.

---

## Nodos Incluidos

Una vez instalado, encontrarás los nuevos nodos bajo la categoría **AudioExtract** al hacer clic derecho en el lienzo de ComfyUI.

### 1. Load Simple Audio (Audio/Video)
Carga un único archivo de audio o video.
* **Entrada**: `audio_path` (Ruta absoluta o relativa del archivo).
* **Salida**: `AUDIO` (Tensor de audio y Sample Rate compatible con ComfyUI).

### 2. Load Folder Audio (Batch)
Carga archivos secuencialmente desde un directorio. Ideal para conectar el índice (`index`) a un nodo que auto-incremente su valor por cada generación del lote.
* **Entradas**:
  * `folder_path`: Ruta a la carpeta que contiene los medios.
  * `index`: Índice del archivo a procesar (0, 1, 2...). Vuelve al inicio automáticamente si el índice supera la cantidad de archivos.
* **Salidas**:
  * `AUDIO`: El flujo de audio procesado.
  * `STRING`: Ruta absoluta del archivo actualmente cargado.

### 3. UVR5 Extractor (Ultimate Vocal Remover)
El núcleo de la separación. Usa el motor MDX-Net para extraer la voz limpia.
* **Entradas**:
  * `audio`: El cable de audio que proviene de los nodos de carga descritos arriba.
  * `reference_audio` *(Opcional)*: Un cable preparado para futuras implementaciones de Ensemble Mode o comparación de fase/ruido. Puedes dejarlo desconectado o conectarle un archivo instrumental si experimentas con sustracción.
* **Configuración**:
  * `model_name`: Modelo a utilizar. Por defecto es `Kim_Vocal_2.onnx`, ampliamente reconocido por su gran calidad al extraer voces de pistas musicales.
* **Salida**:
  * `AUDIO`: El audio resultante (el "Stem" de las voces) limpio y listo para conectarse a un nodo de *Save Audio* o *Preview Audio*.

---

## Guía de Conexión Básica

1. Crea un nodo **Load Simple Audio (Audio/Video)**. Escribe la ruta a un archivo MP4 o WAV.
2. Crea el nodo **UVR5 Extractor**.
3. Conecta el cable amarillo `AUDIO` del nodo de carga hacia la entrada `audio` del Extractor.
4. Crea un nodo estándar de ComfyUI como **SaveAudio** (suele venir con nodos adicionales de audio o ComfyUI nativo) y conecta la salida `AUDIO` del Extractor a este nodo para guardar el resultado.
5. Presiona "Queue Prompt". El nodo descargará automáticamente el modelo `Kim_Vocal_2.onnx` en su primera ejecución (si no está cacheado en el sistema) y procederá a separar las voces.
