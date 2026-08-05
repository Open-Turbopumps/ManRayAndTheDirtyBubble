import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import tempfile
import os


def mp4_to_spectrogram(mp4_file, output_png="spectrogram.png"):
    # Create temporary WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_file = tmp.name

    try:
        # Extract audio with ffmpeg
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", mp4_file,
            "-vn",
            "-ac", "1",        # mono
            "-ar", "44100",    # sample rate
            wav_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read WAV
        sample_rate, audio = wavfile.read(wav_file)

        # Convert to float if needed
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)
            audio /= np.iinfo(np.int16).max

        # Compute spectrogram
        f, t, Sxx = spectrogram(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=2048,
            noverlap=1536,
            scaling="density",
            mode="magnitude"
        )

        # Convert to dB
        Sxx_db = 20 * np.log10(Sxx + 1e-10)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.yscale("log")
        plt.colorbar(label="Magnitude (dB)")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close()

        print(f"Saved to {output_png}")

    finally:
        os.remove(wav_file)


if __name__ == "__main__":
    mp4_to_spectrogram("input.mp4")