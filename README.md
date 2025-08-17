![Warlock-Studio banner](Assets/banner.png)

<div align="center">

### _AI Media Enhancement Suite_

[![Build Status](https://img.shields.io/badge/build-Stable_Release-blue?style=for-the-badge)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![Version](https://img.shields.io/badge/Version-4.1--08.01-darkred?style=for-the-badge)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases/tag/4.1)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Downloads](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&color=gold)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)

_Transform your media with cutting-edge AI technology_

---

**Warlock-Studio** is an open-source desktop application for **Windows** that integrates state-of-the-art AI models for video and image enhancement.  
Inspired by [Djdefrag](https://github.com/Djdefrag) tools like **QualityScaler** and **FluidFrames**, this suite offers a unified, high-performance interface for **upscaling, restoration, and frame interpolation**.

Version **4.1** introduces improved GPU utilization, compatibility fixes, and optimized model loading for a faster, more stable experience.

---

## 📥 Download Installer (v4.1)

Get the latest stable release:

<table>
  <tr>
    <td align="center">
      <a href="https://sourceforge.net/p/warlock-studio/">
        <img alt="Download Warlock-Studio" src="https://sourceforge.net/sflogo.php?type=18&amp;group_id=3880091" width="200">
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v4.1/Warlock-Studio4.1Setup.zip">
        <img src="rsc/GitHub_Logo_WS.png" alt="Download from GitHub" width="200" />
      </a>
    </td>
    <td align="center">
      <a href="https://ivanayub97.itch.io/warlock-studio">
        <img src="rsc/itch.io.png" alt="Download from itch.io" width="200" />
      </a>
    </td>
  </tr>
</table>

---

## ✨ Key Features

- **AI Upscaling & Restoration** – Real-ESRGAN, BSRGAN, IRCNN for denoising, upscaling, and detail recovery.  
- **Face Restoration (GFPGAN)** – Revive blurry or low-quality portraits in photos and videos.  
- **Frame Interpolation (RIFE)** – Generate **2×, 4×, 8×** smoother motion or slow-motion.  
- **Modern UI** – Redesigned in v4.0 for an intuitive, streamlined experience.  
- **Batch Processing** – Handle multiple media files simultaneously.  
- **Custom Workflows** – Full control over models, resolution, output format, and quality.  
- **Open-Source & Extensible** – Licensed under MIT, with a modular architecture for contributors.  

---

## 🆕 What’s New in v4.1

- 🔧 Removed outdated SuperResolution-10 model.  
- ✅ Robust ONNX loading & GPU acceleration.  
- ✅ Fixed import errors & improved type annotations.  
- ✅ Enhanced error handling with graceful fallbacks.  
- 🟢 Better GPU utilization & resource management.  
- 🚀 Compatibility fixes for NumPy & OpenCV.  
- 📦 Stability improvements & refined memory usage.  
- ✅ Improved startup reliability & user notifications.  

---

## 🌐 Smart Model Distribution System (v4.0+)

### 🎯 Lightweight Installation
- Installer reduced from **1.4GB → 450MB** (–68%).  
- Models (~400MB) download automatically on first launch.  
- Bandwidth-friendly setup.  

### 🛡️ Reliability
- **Integrity checks** on downloaded models.  
- **Graceful degradation** if models are missing.  
- **Offline support** for manual model placement.  

---

## 🖼️ Interface Previews

**Main Window**  
![Main interface](rsc/Capture.png)

**RIFE Options**  
![RIFE Options](rsc/CaptureRIFE.png)

---

## 🚀 How to Use

1. Run as **Administrator** (recommended).  
2. **Load Media**: Import images or videos.  
3. **Configure Settings**:  
   - Choose AI model (Real-ESRGAN, GFPGAN, etc.)  
   - Set resolution, format, interpolation, etc.  
4. **Start Processing** with **"Make Magic"**.  
5. Retrieve results from the output folder.  

---

## 🖼️ Quality Comparison

Enhanced image using **BSRGANx2**  
![Comparison](rsc/image_comparison.png)

---

## 📊 Model Comparison

| Model File              | Use Case                                | Speed   | Quality | Notes |
|--------------------------|------------------------------------------|---------|---------|-------|
| **GFPGANv1.4**          | Face restoration                        | High    | High    | Great for blurry faces |
| **BSRGANx2**            | 2× upscale + denoising                  | Medium  | Very High | For lightly degraded images |
| **BSRGANx4**            | 4× upscale + denoising                  | Low     | Very High | For heavily degraded media |
| **RIFE**                | Smooth frame interpolation              | High    | High    | Best quality for motion |
| **RIFE-Lite**           | Faster interpolation                    | Very High | Medium  | Lightweight alternative |
| **RealESRGANx4**        | General 4× upscale                      | Medium  | High    | Great all-rounder |
| **RealESRNetx4**        | Subtle restoration without oversharpen  | Medium  | High    | Preserves natural look |
| **RealSRx4_Anime**      | Anime / line-art upscale                | Medium  | High    | Clean edges for 2D art |
| **IRCNN_L**             | Light denoising                         | High    | Medium  | Mild artifact removal |
| **IRCNN_M**             | Medium denoising                        | High    | Medium  | Stronger artifact cleanup |

---

## ⚙️ Installation

1. **Download installer** (links above).  
2. **Run setup** and follow steps.  
3. Launch from Start Menu / Desktop shortcut.  

Warlock-Studio is packaged with **PyInstaller** + **Inno Setup**.

### Installer Previews

![Installer 1](rsc/Installation_window.png)  
![Installer 2](rsc/Installation_window2.png)  
![Installer 3](rsc/Installation_window3.png)  

---

## 🖥️ System Requirements

- **OS:** Windows 10+ (64-bit)  
- **RAM:** 8GB+ recommended  
- **GPU:** NVIDIA or DirectML-compatible GPU recommended  
- **Storage:** Enough for input + processed media  

---

## 📌 Development Status (v4.1-08.01)

| Component                  | Status      | Notes |
|-----------------------------|------------|-------|
| Upscaling Models            | 🟢 Stable  | VRAM recovery integrated |
| Optimized Model Suite       | 🟢 Enhanced | Streamlined & reliable |
| Face Restoration (GFPGAN)   | 🟢 Stable  | High-quality face fix |
| Frame Interpolation (RIFE)  | 🟢 Stable  | Smooth motion, slow-mo |
| Batch Processing            | 🟢 Stable  | Improved error handling |
| User Interface (UI/UX)      | 🟢 Refined | Clean, integrated models |
| GPU Management              | 🟢 Enhanced | Robust ONNX + fallbacks |
| Code Quality                | 🟢 Improved | Refactored & type-safe |
| Installer & Packaging       | 🟢 Stable  | Seamless setup |

---

## 📂 Project Structure
</div>

```bash
Warlock-Studio/
├── AI-onnx/                          # Pre-trained ONNX models for AI processing
│   ├── BSRGANx2_fp16.onnx
│   ├── BSRGANx4_fp16.onnx
│   ├── GFPGANv1.4.fp16.onnx
│   ├── IRCNN_Lx1_fp16.onnx
│   ├── IRCNN_Mx1_fp16.onnx
│   ├── RealESR_Animex4_fp16.onnx
│   ├── RealESR_Gx4_fp16.onnx
│   ├── RealESRGANx4_fp16.onnx
│   ├── RealESRNetx4_fp16.onnx
│   ├── RealSRx4_Anime_fp16.onnx
│   ├── RIFE_fp32.onnx
│   └── RIFE_Lite_fp32.onnx
│
├── Assets/                           # App assets and third-party binaries
│   ├── banner.png
│   ├── clear_icon.png
│   ├── exiftool.exe
│   ├── ffmpeg.exe
│   ├── info_icon.png
│   ├── logo.ico
│   ├── logo.png
│   ├── stop_icon.png
│   ├── upscale_icon.png
│   ├── wizard-image.bmp
│   └── wizard-small.bmp
│
├── rsc/                              # UI images, workflows, and branding
│   ├── badge-color.png
│   ├── Capture.png
│   ├── CaptureRIFE.png
│   ├── google_drive-logo.png
│   ├── WorkflowBSRGAN.png
│   ├── WorkflowIRCNN.png
│   ├── WorkflowRealESRGAN.png
│   ├── WorkflowRIFE.png
│   └── Installation_window2.png
│
├── Manual/                           # LaTeX sources and generated manuals
│   ├── Manual_EN.tex
│   ├── Manual_ES.tex
│   ├── Manual_EN.pdf
│   └── Manual_ES.pdf
│
├── Warlock-Studio.py                 # Main application script
├── Warlock-Studio.spec               # PyInstaller spec file
├── Setup.iss                         # Inno Setup installer script
│
├── README.md                         # Project overview (this file)
├── requirements.txt                
├── CHANGELOG.md                      # Version history
├── LICENSE                           # License info (standard)
├── License.txt                   
├── NOTICE.md                         # Notices and attributions
├── CODE_OF_CONDUCT.md                # Contributor behavior guidelines
├── CONTRIBUTING.md                   # Contribution guide
└── SECURITY.md                       # Security reporting policies


```
<div align="center">

---

## 🔗 Integrated Technologies & Licenses

| Technology    | License     | Author | Source |
|---------------|------------|--------|--------|
| Real-ESRGAN   | BSD/Apache | [Xintao Wang](https://github.com/xinntao) | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| GFPGAN        | Apache 2.0 | [TencentARC](https://github.com/TencentARC) | [GitHub](https://github.com/TencentARC/GFPGAN) |
| RIFE          | Apache 2.0 | [hzwer](https://github.com/hzwer) | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| BSRGAN        | Apache 2.0 | [Kai Zhang](https://github.com/cszn) | [GitHub](https://github.com/cszn/BSRGAN) |
| IRCNN         | BSD/Mixed  | [Kai Zhang](https://github.com/cszn) | [GitHub](https://github.com/cszn/IRCNN) |
| Anime4K       | MIT        | [bloc97](https://github.com/bloc97) | [GitHub](https://github.com/bloc97/Anime4K) |
| ONNX Runtime  | MIT        | [Microsoft](https://github.com/microsoft) | [GitHub](https://github.com/microsoft/onnxruntime) |
| PyTorch       | BSD-3      | [Meta AI](https://pytorch.org/) | [GitHub](https://github.com/pytorch/pytorch) |
| FFmpeg        | LGPL/GPL   | [FFmpeg Team](https://ffmpeg.org) | [Site](https://ffmpeg.org) |
| ExifTool      | Artistic   | [Phil Harvey](https://exiftool.org/) | [Site](https://exiftool.org/) |
| PyInstaller   | GPLv2+     | [Team](https://github.com/pyinstaller) | [GitHub](https://github.com/pyinstaller/pyinstaller) |
| Inno Setup    | Custom     | [Jordan Russell](http://www.jrsoftware.org/) | [Site](http://www.jrsoftware.org/isinfo.php) |

---

## 🤝 Contributions

We welcome community contributions!  

1. **Fork** the repo  
2. **Create a branch** for your feature/fix  
3. **Submit a Pull Request** with details  

📧 Contact: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**  

---

## 📜 License

© 2025 Iván Eduardo Chavez Ayub  
Licensed under **MIT**. Additional terms in `NOTICE.md`.  
</div>




