![Warlock-Studio banner](Assets/banner.png)
![Build Status](https://img.shields.io/badge/build-Stable_Release-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/%20Version-2.2-darkred?style=for-the-badge)

### Get Warlock-Studio Installer

You can download the installer (latest version **2.2**) from any of this platforms:

<table>
  <tr>
    <td align="center">
      <a href="https://sourceforge.net/projects/warlock-studio/files/latest/download">
        <img src="https://a.fsdn.com/con/app/sf-download-button" alt="Download from SourceForge" />
      </a>
    </td>
    <td align="center">
      <a href="https://drive.google.com/file/d/1nqlBuxZKsk3FX_nWUqUnDUKUTfMSfjB4/view?usp=sharing">
        <img src="rsc/google_drive-logo.png" alt="Download from Google Drive" />
      </a>
    </td>
  </tr>
</table>

### AI-Powered Media Enhancement & Upscaling Suite 2.2

**Warlock-Studio** is a powerful **open-source desktop application** inspired by the remarkable work of [Djdefrag](https://github.com/Djdefrag), integrating tools like **QualityScaler**, **RealScaler**, and **FluidFrames**. Built with performance and usability in mind, Warlock-Studio brings together the best of these technologies into a unified and user-friendly interface.

---

It features integration with state-of-the-art models for upscaling, restoration, and frame interpolation—all within an intuitive and streamlined user interface. Warlock-Studio delivers **professional-grade media processing** capabilities to everyone.

Version 2.2 introduces critical improvements focused on reliability and performance:

- **Comprehensive Logging System** for easier debugging.
- **Proactive Environment Validation** to prevent common errors.
- **Resilient Video Encoding** with automatic codec and audio fallbacks.
- **Aggressive Memory Management** and dynamic GPU VRAM recovery to handle long processing tasks without crashing.

---

## Interface Previews

### 🔹 Main

![Screenshot of Warlock-Studio](rsc/Capture.png)

### 🔹 RIFE (Frame Interpolation) Options

![Screenshot of Warlock-Studio](rsc/CaptureRIFE.png)

### 🔹 Icon App

## ![Screenshot of Warlock-Studio](logo.ico)

### 🔹 Installation Window

## ![Screenshot of Warlock-Studio](rsc/Installation_window.png)

## ![Screenshot of Warlock-Studio](rsc/Installation_window2.png)

## 🛠️ Development Status — v2.2

| Component                           | Status           | Notes                                                                     |
| :---------------------------------- | :--------------- | :------------------------------------------------------------------------ |
| **Upscaling Models (ESRGAN, etc.)** | 🟢 **Stable**    | Fully integrated with dynamic VRAM recovery for enhanced stability.       |
| **Frame Interpolation (RIFE)**      | 🟢 **Stable**    | Includes slow-motion and intermediate frame generation capabilities.      |
| **Batch Processing**                | 🟢 **Stable**    | Reliable processing with improved error handling and resource management. |
| **User Interface (UI/UX)**          | 🟢 **Improved**  | Updated color palette and faster start-up time.                           |
| **GPU Management**                  | 🟢 **Optimized** | Dynamic VRAM error recovery and graceful hardware codec fallbacks.        |
| **Installer and Packaging**         | 🟢 **Stable**    | Easy-to-use installer for Windows platforms.                              |

---

## Recent Enhancements (v2.2)

- ✅ **Stability Overhaul:** Major improvements in error handling, including a comprehensive logging system and proactive environment validation to prevent crashes.
- ✅ **Resilient Video Processing:** The video encoding pipeline now features automatic fallbacks for hardware codecs and audio stream processing, ensuring a valid output file is always created.
- ✅ **Performance and Memory Optimization:** Implemented aggressive memory management to handle large video files without crashing and added dynamic GPU VRAM recovery for tiling-based tasks.
- ✅ **Critical Bug Fixes:** Resolved race conditions in video encoding and GUI status updates, ensuring process integrity and predictable behavior.
- ✅ **Safe Thread Management:** Upgraded to ensure processes are terminated gracefully and system resources are properly cleaned up on exit.

---

## Project Structure

```
Warlock-Studio/
├──AI-onnx/
   │
   └──├──BSRGANx2_fp16.onnx
      ├──BSRGANx4_fp16.onnx
      ├──IRCNN_Lx1_fp16.onnx
      ├──IRCNN_Mx1_fp16.onnx
      ├──RealESR_Animex4_fp16.onnx
      ├──RealESR_Gx4_fp16.onnx
      ├──RealESRGANx4_fp16.onnx
      ├──RealESRNetx4_fp16.onnx
      ├──RealSRx4_Anime_fp16.onnx
      ├──RIFE_fp32.onnx
      └──RIFE_Lite_fp32.onnx
├──Assets/
   │
   └──├──banner.png
      ├──clear_icon.png
      ├──exiftool.exe
      ├──ffmpeg.exe
      ├──ffmplay.exe
      ├──ffmprobe.exe
      ├──info_icon.png
      ├──logo.ico
      ├──logo.png
      ├──stop_icon.png
      ├──upscale_icon.png
      ├──wizard-image.bmp
      └──wizard-small.bmp
├──rsc/
   │
   └──├──badge-color.png
      ├──Capture.png
      ├──CaptureRIFE.png
      ├──google_drive-logo.png
      ├──Installation_window.png
      └──Installation_window2.png
│
├──Manual/
   │
   └──├──Manual_EN.pdf
      ├──Manual_EN.tex
      ├──Manual_ES.pdf
      └──Manual_ES.tex
│
├──CHANGELOG.md
├──CODE_OF_CONDUCT.md
├──CONTRIBUTING.md
├──LICENSE
├──License.txt
├──NOTICE.md
├──README.md                 # This File
├──SECURITY.md
├──Setup.iss
├──Manual_ES.pdf
├──Manual_EN.pdf
├──Warlock-Studio.py         # Main
└──Warlock-Studio.spec

```

## Installation

To get started with Warlock-Studio:

1. **Run the installer** and follow the setup instructions.
2. **Launch the application** by opening `Warlock-Studio.exe`.
3. **Begin enhancing** your images and videos with just a few clicks\!

Warlock-Studio uses [PyInstaller](https://www.pyinstaller.org/) and [Inno Setup](http://www.jrsoftware.org/isinfo.php) for a seamless packaging and installation experience.

## Key Features

- **State-of-the-Art AI Models**
  Real-ESRGAN, SRGAN, BSRGAN, IRCNN, Waifu2x, Anime4K, **RIFE**, and others for denoising, resolution enhancement, detail restoration, and smooth frame interpolation.

- **AI Frame Interpolation & Slow Motion**
  Generate new in-between frames using RIFE to create smooth **2x/4x/8x** motion or dramatic slow-motion effects.

- **Batch Processing**
  Simultaneously process multiple images or videos—ideal for large-scale media projects.

- **Customizable Workflows**
  Choose your preferred AI model, output resolution, format (PNG, JPEG, MP4, etc.), and quality settings for full creative control.

- **Intuitive Interface**
  Designed for both beginners and professionals—simple, clean, and efficient.

- **Open-Source & Extensible**
  Licensed under the MIT License. Additional usage terms can be found in the [NOTICE](https://www.google.com/search?q=NOTICE.md) file.

---

## How to Use

1. **Run as Administrator** (optional but recommended for optimal performance).

2. **Load your media**: select your images and videos into the app.

3. **Configure settings**:

   - Select an **AI Model** (e.g., Real-ESRGAN, SRGAN, BSRGAN, IRCNN, Waifu2x, Anime4K, RIFE)
   - Set the **output resolution**, **file format**, and toggle features such as **interpolation** or **slow-motion**

4. **Start Processing**: click **"Make Magic"** to begin enhancement.

5. **Retrieve your files**: processed outputs will be saved in your chosen destination folder.

---

## Quality Comparison

**Comparison of an enhanced image using the BSRGANx2 model**
![Quality Comparison](rsc/image_comparison.png)

---

## System Requirements

- **Operating System:** Windows 10 or later
- **Memory (RAM):** Minimum 4 GB (8 GB or more recommended)
- **Graphics Card:** NVIDIA or DirectML-compatible GPU highly recommended for performance
- **Storage:** Sufficient disk space for input and output media files

---

### Integrated Technologies & Licenses

| Technology    | License                          | Author / Maintainer                                     | Source Code / Homepage                                     |
| :------------ | :------------------------------- | :------------------------------------------------------ | :--------------------------------------------------------- |
| QualityScaler | MIT                              | [Djdefrag](https://github.com/Djdefrag)                 | [GitHub](https://github.com/Djdefrag/QualityScaler)        |
| RealScaler    | MIT                              | [Djdefrag](https://github.com/Djdefrag)                 | [GitHub](https://github.com/Djdefrag/RealScaler)           |
| FluidFrames   | MIT                              | [Djdefrag](https://github.com/Djdefrag)                 | [GitHub](https://github.com/Djdefrag/FluidFrames)          |
| Real-ESRGAN   | BSD 3-Clause / Apache 2.0        | [Xintao Wang](https://github.com/xinntao)               | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| RealESRGAN-G  | BSD 3-Clause / Apache 2.0        | [Xintao Wang](https://github.com/xinntao)               | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| RealESR-Anime | BSD 3-Clause / Apache 2.0        | [Xintao Wang](https://github.com/xinntao)               | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| RealESR-Net   | BSD 3-Clause / Apache 2.0        | [Xintao Wang](https://github.com/xinntao)               | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| RIFE          | Apache 2.0                       | [hzwer](https://github.com/hzwer)                       | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| SRGAN         | CC BY-NC-SA 4.0 (Non-Commercial) | [TensorLayer Community](https://github.com/tensorlayer) | [GitHub](https://github.com/tensorlayer/srgan)             |
| BSRGAN        | Apache 2.0                       | [Kai Zhang](https://github.com/cszn)                    | [GitHub](https://github.com/cszn/BSRGAN)                   |
| IRCNN         | BSD / Mixed                      | [Kai Zhang](https://github.com/cszn)                    | [GitHub](https://github.com/cszn/IRCNN)                    |
| Anime4K       | MIT                              | [Tianyang Zhang (bloc97)](https://github.com/bloc97)    | [GitHub](https://github.com/bloc97/Anime4K)                |
| ONNX Runtime  | MIT                              | [Microsoft](https://github.com/microsoft)               | [GitHub](https://github.com/microsoft/onnxruntime)         |
| PyTorch       | BSD 3-Clause                     | [Meta AI](https://pytorch.org/)                         | [GitHub](https://github.com/pytorch/pytorch)               |
| FFmpeg        | LGPL-2.1 / GPL (varies)          | [FFmpeg Team](https://ffmpeg.org/)                      | [Official Site](https://ffmpeg.org)                        |
| ExifTool      | Perl Artistic License 1.0        | [Phil Harvey](https://exiftool.org/)                    | [Official Site](https://exiftool.org/)                     |
| DirectML      | MIT                              | [Microsoft](https://github.com/microsoft/)              | [GitHub](https://github.com/microsoft/DirectML)            |
| Python        | Python Software Foundation (PSF) | [Python Software Foundation](https://www.python.org/)   | [Official Site](https://www.python.org)                    |
| PyInstaller   | GPLv2+                           | [PyInstaller Team](https://github.com/pyinstaller)      | [GitHub](https://github.com/pyinstaller/pyinstaller)       |
| Inno Setup    | Custom Inno License              | [Jordan Russell](http://www.jrsoftware.org/)            | [Official Site](http://www.jrsoftware.org/isinfo.php)      |

---

## Contributions

We warmly welcome community contributions\!

1. **Fork** this repository.
2. **Create a branch** for your feature or fix.
3. **Submit a Pull Request** with a detailed explanation of your changes.

For bug reports, feature suggestions, or inquiries, contact us at: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**

**Warlock-Studio** merges cutting-edge artificial intelligence with a powerful yet accessible interface—empowering creators to elevate their media effortlessly.

---

## License

© 2025 Iván Eduardo Chavez Ayub
Distributed under the MIT License. Additional terms are available in the [NOTICE.md](https://www.google.com/search?q=NOTICE.md) file.
