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

# Supported formats
AUDIO_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma')
VIDEO_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
ALL_FORMATS = AUDIO_FORMATS + VIDEO_FORMATS

def load_audio_or_video(file_path):
    """
    Loads an audio or video file, extracts the audio, and returns a ComfyUI compatible audio dictionary.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in VIDEO_FORMATS:
        # Use moviepy to extract audio from video
        clip = AudioFileClip(file_path)
        sample_rate = int(clip.fps)
        # Get audio array (shape: [num_frames, num_channels] or [num_frames])
        audio_array = clip.to_soundarray()
        clip.close()

        if audio_array.ndim == 1:
            # Mono to [channels, frames]
            audio_array = audio_array.reshape(1, -1)
        else:
            # Stereo/multi-channel to [channels, frames]
            audio_array = audio_array.T

        # Convert to float32 tensor
        waveform = torch.from_numpy(audio_array).to(torch.float32)

    elif ext in AUDIO_FORMATS:
        # Load directly with soundfile
        try:
            audio_array, sample_rate = sf.read(file_path)
            if audio_array.ndim == 1:
                # Mono to [channels, frames]
                audio_array = audio_array.reshape(1, -1)
            else:
                # Stereo/multi-channel to [channels, frames]
                audio_array = audio_array.T
            waveform = torch.from_numpy(audio_array).to(torch.float32)
        except Exception as e:
            # Fallback a moviepy si soundfile falla (ej. algunos mp3/m4a)
            clip = AudioFileClip(file_path)
            sample_rate = int(clip.fps)
            audio_array = clip.to_soundarray()
            clip.close()
            if audio_array.ndim == 1:
                audio_array = audio_array.reshape(1, -1)
            else:
                audio_array = audio_array.T
            waveform = torch.from_numpy(audio_array).to(torch.float32)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

class LoadSimpleAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio_path": ("STRING", {"default": "path/to/audio/or/video.mp4"})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "AudioExtract"

    def load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        audio_dict = load_audio_or_video(audio_path)
        return (audio_dict,)

class LoadFolderAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "path/to/folder"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999})
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    FUNCTION = "load_batch"
    CATEGORY = "AudioExtract"

    def load_batch(self, folder_path, index):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(ALL_FORMATS)]
        files.sort()

        if not files:
            raise ValueError(f"No audio/video files found in {folder_path}.")

        selected_file = os.path.join(folder_path, files[index % len(files)])
        audio_dict = load_audio_or_video(selected_file)

        return (audio_dict, selected_file)

class UVRExtractorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (["Kim_Vocal_2.onnx", "UVR-MDX-NET-Voc-FT.onnx"], {"default": "Kim_Vocal_2.onnx"}),
            }
        }

    # Definimos dos salidas claras para el workflow
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("VOCALS", "INSTRUMENTAL")
    FUNCTION = "separate"
    CATEGORY = "AudioExtract"

    def separate(self, audio, model_name="Kim_Vocal_2.onnx"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input_audio.wav")
            waveform = audio["waveform"].squeeze(0)
            sample_rate = audio["sample_rate"]

            # soundfile espera [frames, channels]
            audio_np = waveform.numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T
            sf.write(input_path, audio_np, sample_rate)

            # Inicializar separador UVR5
            separator = Separator(output_dir=tmpdir)
            separator.load_model(model_name)
            output_files = separator.separate(input_path)

            vocal_path = None
            inst_path = None

            # Lógica de identificación robusta
            if len(output_files) >= 2:
                for file in output_files:
                    full_path = os.path.join(tmpdir, file)
                    if "vocal" in file.lower():
                        vocal_path = full_path
                    else:
                        inst_path = full_path # El que no es vocal, es instrumental

            # Fallback en caso de nombres inesperados o un solo archivo
            if not vocal_path:
                vocal_path = os.path.join(tmpdir, output_files[0])
            if not inst_path and len(output_files) > 1:
                inst_path = os.path.join(tmpdir, output_files[1])

            # Carga de resultados con soundfile
            v_array, v_sr = sf.read(vocal_path)
            if v_array.ndim == 1:
                v_array = v_array.reshape(1, -1)
            else:
                v_array = v_array.T
            v_wave = torch.from_numpy(v_array).to(torch.float32)

            # Si no hay instrumental (raro), devolvemos silencio para no romper el flujo
            if inst_path:
                i_array, i_sr = sf.read(inst_path)
                if i_array.ndim == 1:
                    i_array = i_array.reshape(1, -1)
                else:
                    i_array = i_array.T
                i_wave = torch.from_numpy(i_array).to(torch.float32)
            else:
                i_wave, i_sr = torch.zeros_like(v_wave), v_sr

            return (
                {"waveform": v_wave.unsqueeze(0), "sample_rate": v_sr},
                {"waveform": i_wave.unsqueeze(0), "sample_rate": i_sr}
            )


NODE_CLASS_MAPPINGS = {
    "LoadSimpleAudio": LoadSimpleAudio,
    "LoadFolderAudio": LoadFolderAudio,
    "UVRExtractor": UVRExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSimpleAudio": "Load Simple Audio (Audio/Video)",
    "LoadFolderAudio": "Load Folder Audio (Batch)",
    "UVRExtractor": "Extractor UVR5 (Vocal + Instrumental)"
}
