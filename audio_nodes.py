import os
import tempfile
import torch
import numpy as np
import torchaudio
from moviepy.editor import AudioFileClip
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
        # Load directly with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
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
            },
            "optional": {
                "reference_audio": ("AUDIO",), # Optional reference audio input for phase cancellation or noise profile
                "model_name": (["Kim_Vocal_2.onnx", "UVR-MDX-NET-Voc-FT.onnx"], {"default": "Kim_Vocal_2.onnx"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "separate"
    CATEGORY = "AudioExtract"

    def separate(self, audio, reference_audio=None, model_name="Kim_Vocal_2.onnx"):
        # The separator expects a file path. We use a temporary directory to handle concurrent executions safely.
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input_audio.wav")

            # ComfyUI audio format is typically: dict with "waveform" and "sample_rate"
            waveform = audio["waveform"].squeeze(0) # [channels, frames]
            sample_rate = audio["sample_rate"]

            # Save the waveform to a temporary wav file
            torchaudio.save(input_path, waveform, sample_rate)

            # Initialize separator
            separator = Separator(output_dir=tmpdir)
            separator.load_model(model_name)

            # Note: The reference_audio is kept here as a placeholder for potential Ensemble Mode
            # or advanced phase cancellation features as requested. Currently audio-separator
            # only takes a single input path in its basic `separate` method.
            if reference_audio is not None:
                # Potential logic to utilize reference_audio as a noise profile or bias could go here
                # Example: reference_path = os.path.join(tmpdir, "reference_audio.wav")
                # torchaudio.save(reference_path, reference_audio["waveform"].squeeze(0), reference_audio["sample_rate"])
                pass

            # Perform separation
            output_files = separator.separate(input_path)

            # Dynamically search for the Vocals stem
            vocal_file = None
            for file in output_files:
                # audio-separator generally appends the stem name, e.g., (Vocals).wav
                if "Vocal" in file or "vocal" in file:
                    vocal_file = file
                    break

            if vocal_file is None:
                # Fallback to the first output if we can't identify by name
                vocal_file = output_files[0] if output_files else None

            if not vocal_file:
                raise RuntimeError("UVR5 separation failed, no output files generated.")

            vocal_path = os.path.join(tmpdir, vocal_file)

            # Load the result (Vocal)
            clean_waveform, sr = torchaudio.load(vocal_path)

            return ({"waveform": clean_waveform.unsqueeze(0), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "LoadSimpleAudio": LoadSimpleAudio,
    "LoadFolderAudio": LoadFolderAudio,
    "UVRExtractor": UVRExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSimpleAudio": "Load Simple Audio (Audio/Video)",
    "LoadFolderAudio": "Load Folder Audio (Batch)",
    "UVRExtractor": "UVR5 Extractor (Ultimate Vocal Remover)"
}
