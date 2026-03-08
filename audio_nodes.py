import os
import tempfile
import torch
import numpy as np
import soundfile as sf
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip
from audio_separator.separator import Separator

AUDIO_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma')
VIDEO_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
ALL_FORMATS = AUDIO_FORMATS + VIDEO_FORMATS

def load_audio_or_video(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in VIDEO_FORMATS:
        clip = AudioFileClip(file_path)
        sample_rate = int(clip.fps)
        audio_array = clip.to_soundarray()
        clip.close()
        # Moviepy devuelve [frames, channels], pasamos a [channels, frames]
        waveform = torch.from_numpy(audio_array.T).to(torch.float32)
    elif ext in AUDIO_FORMATS:
        data, sample_rate = sf.read(file_path)
        # Aseguramos [channels, frames]
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0).to(torch.float32)
        else:
            waveform = torch.from_numpy(data.T).to(torch.float32)

    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

class UVRExtractorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": ([
                    "Kim_Vocal_2.onnx",
                    "UVR-MDX-NET-Voc_FT.onnx",
                    "UVR_MDX_NET_Main.onnx",
                    "UVR-MDX-NET-Inst_Main.onnx"
                ], {"default": "Kim_Vocal_2.onnx"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("VOCALS", "INSTRUMENTAL")
    FUNCTION = "separate"
    CATEGORY = "AudioExtract"

    def separate(self, audio, model_name="Kim_Vocal_2.onnx"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input_audio.wav")
            waveform = audio["waveform"].squeeze(0) # [channels, frames]
            sample_rate = audio["sample_rate"]

            # Guardar usando soundfile (requiere [frames, channels])
            sf.write(input_path, waveform.cpu().numpy().T, sample_rate)

            separator = Separator(output_dir=tmpdir)

            # Lógica de carga con Fallback
            try:
                print(f"UVR5: Cargando modelo {model_name}...")
                separator.load_model(model_name)
            except Exception as e:
                print(f"UVR5 Error: El modelo {model_name} no es válido o falló. Usando Kim_Vocal_2 como respaldo. Error: {e}")
                # Descomenta la siguiente línea si necesitas ver todos los modelos disponibles:
                # print(f"Modelos disponibles: {separator.list_supported_model_files()}")
                separator.load_model("Kim_Vocal_2.onnx")

            output_files = separator.separate(input_path)

            vocal_path = None
            inst_path = None

            for file in output_files:
                full_path = os.path.join(tmpdir, file)
                if "vocal" in file.lower():
                    vocal_path = full_path
                else:
                    inst_path = full_path

            # Si el modelo no separa (raro), el primer archivo es la voz
            if not vocal_path and output_files:
                vocal_path = os.path.join(tmpdir, output_files[0])

            # Cargar resultados con soundfile
            v_data, v_sr = sf.read(vocal_path)
            v_wave = torch.from_numpy(v_data.T).to(torch.float32) if v_data.ndim > 1 else torch.from_numpy(v_data).unsqueeze(0).to(torch.float32)

            if inst_path:
                i_data, i_sr = sf.read(inst_path)
                i_wave = torch.from_numpy(i_data.T).to(torch.float32) if i_data.ndim > 1 else torch.from_numpy(i_data).unsqueeze(0).to(torch.float32)
            else:
                i_wave = torch.zeros_like(v_wave)
                i_sr = v_sr

            return (
                {"waveform": v_wave.unsqueeze(0), "sample_rate": v_sr},
                {"waveform": i_wave.unsqueeze(0), "sample_rate": i_sr}
            )

class LoadSimpleAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio_path": ("STRING", {"default": "path/to/file.mp4"})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    CATEGORY = "AudioExtract"

    def load(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"No se encuentra: {audio_path}")
        return (load_audio_or_video(audio_path),)

NODE_CLASS_MAPPINGS = {
    "LoadSimpleAudio": LoadSimpleAudio,
    "UVRExtractor": UVRExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSimpleAudio": "Cargar Audio/Video Simple",
    "UVRExtractor": "Extractor UVR5 PRO"
}
