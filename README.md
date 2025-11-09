![Warlock-Studio banner](Assets/banner.png)

<div align="center">

[![Build Status](https://img.shields.io/badge/Build-Stable_Release-0A192F?style=for-the-badge&logo=github&logoColor=FFD700)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![Version](https://img.shields.io/badge/Version-4.3-FF4500?style=for-the-badge&logo=git&logoColor=white)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases/tag/4.3)
[![License](https://img.shields.io/badge/License-MIT-6A0DAD?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Downloads](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&color=FFD700&logo=download&logoColor=black)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)

[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?style=for-the-badge&logo=windows11&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=FFD700)](https://www.python.org/)
[![Issues](https://img.shields.io/github/issues/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=FF0303&logo=github&logoColor=white)](https://github.com/Ivan-Ayub97/Warlock-Studio/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=2E8B57&logo=git&logoColor=white)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)

Inspired by [Djdefrag](https://github.com/Djdefrag) tools such as **QualityScaler** and **FluidFrames**, Warlock-Studio provides a unified, high-performance platform for **upscaling, restoration, denoising, and frame interpolation**.

---

`Fixed missing audio in generated videos.`  
`Corrected dark output issue in GFPGAN face restoration.`  
`Improved overall stability and synchronization.`  

See CHANGELOG.md for full details.

---

## 📥 **Download Installer** 
from any of this options:
<table style="width:100%; border-collapse:collapse;">
  <tr>
    <td align="center" style="vertical-align:top; padding:px;">
      <a href="https://sourceforge.net/projects/warlock-studio/" target="_blank">
        <img src="https://sourceforge.net/cdn/syndication/badge_img/3880091/oss-rising-star-black"
             alt="Warlock-Studio on SourceForge"
             width="175" style="display:block; margin:auto; margin-bottom:1px;" />
      </a>
    </td>
    <td align="center" style="vertical-align:top; padding:10px;">
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v4.3/Warlock-Studio-v4.3-Setup.zip" target="_blank">
        <img src="rsc/GitHub_Logo_WS.png" alt="Download from GitHub"
             width="300" style="display:block; margin:auto; margin-bottom:10px;" />
      </a>
    </td>
  </tr>
</table>

---

## 🖼️ Interface Previews

![Main interface](rsc/Capture.png)

---

## 🖼️ Quality Comparison

[WsvideovsGit.webm](https://github.com/user-attachments/assets/c72f389d-827e-49b9-91b7-fd13e5b59f22)

[WsvideovsGit2.webm](https://github.com/user-attachments/assets/6695cce2-f42f-4955-8b43-56ec6d7b0bd2)

![Comparison](rsc/image_comparison.png)

---

## ✨ Key Features

- **AI Upscaling & Restoration** – Utilize **Real-ESRGAN, BSRGAN, and IRCNN** models for denoising, super-resolution, and detail recovery.
- **Face Restoration (GFPGAN)** – Recover facial details from low-resolution or blurry images and video frames.
- **Frame Interpolation (RIFE)** – Smooth motion or generate slow-motion content with **2×, 4×, or 8× interpolation**.
- **Advanced Hardware Acceleration** – Intelligent provider selection prioritizes **CUDA**, falls back to **DirectML**, and finally **CPU** for maximum compatibility and performance.
- **Batch Processing** – Process multiple media files simultaneously, saving time and effort.
- **Custom Workflows** – Fine-grained control over models, resolution, output formats, and quality parameters.
- **Open-Source & Extensible** – Fully MIT licensed, for contributors and developers.

---

## 🖥️ System Requirements

- **OS:** Windows 11 or higher (64-bit)
- **RAM:** 8GB+ recommended
- **GPU:** NVIDIA (for CUDA), AMD, or Intel GPU with up-to-date drivers recommended
- **Storage:** Sufficient free space for input and processed media

---

## 📊 Model Comparison

| Model File         | Use Case                     | Speed     | Quality   | Notes                               |
| :----------------- | :--------------------------- | :-------- | :-------- | :---------------------------------- |
| **GFPGANv1.4**     | Face restoration             | High      | High      | Optimal for portraits               |
| **BSRGANx2**       | 2× upscale + denoising       | Medium    | Very High | Suitable for lightly degraded media |
| **BSRGANx4**       | 4× upscale + denoising       | Low       | Very High | For heavily degraded content        |
| **RIFE**           | Frame interpolation          | High      | High      | Smooth motion, slow-motion support  |
| **RIFE-Lite**      | Lightweight interpolation    | Very High | Medium    | Faster, lower resource usage        |
| **RealESRGANx4**   | General 4× upscaling         | Medium    | High      | Balanced performance                |
| **RealESRNetx4**   | Subtle restoration           | Medium    | High      | Preserves natural image texture     |
| **RealSRx4_Anime** | Anime / line-art enhancement | Medium    | High      | Sharp edges for 2D art              |
| **IRCNN_L**        | Light denoising              | High      | Medium    | Mild artifact removal               |
| **IRCNN_M**        | Medium denoising             | High      | Medium    | Stronger artifact cleanup           |

---

## 📂 Project Structure

</div>

```
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
├── Assets/                           # Application assets and third-party binaries
│   ├── banner.png
│   ├── clear_icon.png
│   ├── exiftool.exe
│   ├── ffmpeg.exe
│   ├── ffmprobe.exe
│   ├── ffmplay.exe
│   ├── info_icon.png
│   ├── logo.ico
│   ├── logo.png
│   ├── stop_icon.png
│   ├── upscale_icon.png
│   ├── wizard-image.bmp
│   └── wizard-small.bmp
│
├── rsc/                              # UI previews and branding resources
│   ├── Capture.png
│   ├── image_comparison.png
│   ├── CaptureCONSOLE.png
│   └── GitHub_Logo_WS.png
│
├── Warlock-Studio.py                 # Main application script
├── Warlock-Studio.spec               # PyInstaller specification file
├── Setup.iss                         # Inno Setup installer script
├── README.md                         # Project overview
├── CHANGELOG.md                      # Version history and updates
├── LICENSE                           # MIT License information
├── NOTICE.md                         # Legal notices and attributions
├── CODE_OF_CONDUCT.md                # Contributor guidelines
├── CONTRIBUTING.md                   # Contribution guide
└── SECURITY.md                       # Security reporting policies
```

<div align="center">

---

## 🤝 Contributions

We welcome contributions from the community:

**Fork** the repository.
**Create a branch** for your feature or bug fix.
**Submit a Pull Request** with a detailed description and testing notes.

📧 Contact: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**

---

## 📜 License

© 2025 Iván Eduardo Chavez Ayub
Licensed under **MIT**. Additional terms and attributions are provided in `NOTICE.md`.

### 📊 Integrated Technologies & Licenses

| Technology    | License               | Author / Maintainer                     | Source                                                     |
| ------------- | --------------------- | --------------------------------------- | ---------------------------------------------------------- |
| QualityScaler | MIT                   | [Djdefrag](https://github.com/Djdefrag) | [GitHub](https://github.com/Djdefrag/QualityScaler)        |
| FluidFrames   | MIT                   | [Djdefrag](https://github.com/Djdefrag) | [GitHub](https://github.com/Djdefrag/FluidFrames)          |
| Real-ESRGAN   | BSD 3-Clause / Apache | Xintao Wang                             | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| GFPGAN        | Apache 2.0            | TencentARC / Xintao Wang                | [GitHub](https://github.com/TencentARC/GFPGAN)             |
| RIFE          | Apache 2.0            | hzwer                                   | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| BSRGAN        | Apache 2.0            | Kai Zhang                               | [GitHub](https://github.com/cszn/BSRGAN)                   |
| IRCNN         | BSD / Mixed           | Kai Zhang                               | [GitHub](https://github.com/cszn/IRCNN)                    |
| ONNX Runtime  | MIT                   | Microsoft                               | [GitHub](https://github.com/microsoft/onnxruntime)         |
| FFmpeg        | LGPL / GPL            | FFmpeg Team                             | [Official Site](https://ffmpeg.org)                        |
| ExifTool      | Artistic License      | Phil Harvey                             | [Official Site](https://exiftool.org/)                     |
| Python        | PSF License           | Python Software Foundation              | [Official Site](https://www.python.org)                    |
| PyInstaller   | GPLv2+                | PyInstaller Team                        | [GitHub](https://github.com/pyinstaller/pyinstaller)       |
| Inno Setup    | Custom                | Jordan Russell                          | [Official Site](http://www.jrsoftware.org/isinfo.php)      |

</div>















