![Warlock-Studio banner](Assets/banner.png)

<div align="center">

![Platform](https://img.shields.io/badge/Platform-Windows_10_%7C_11-003B6F?style=for-the-badge&logo=windows&logoColor=FFD43B&labelColor=1B1818)
![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=FFD43B&labelColor=1B1818)
![License: MIT](https://img.shields.io/badge/License-MIT-0A3B1E?style=for-the-badge&logo=open-source-initiative&logoColor=FFD43B&labelColor=1B1818)

[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&logo=git&color=6A1B9A&logoColor=FFD43B&labelColor=1B1818)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)
![Version 5.1.1](https://img.shields.io/badge/Version-5.1.1-660066?style=for-the-badge&logo=tag&logoColor=FFD43B&labelColor=1B1818)

[![Downloads Total](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&logo=github&color=2E2E2E&labelColor=1B1818&logoColor=FFD43B)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![SF Downloads](https://img.shields.io/sourceforge/dt/warlock-studio?style=for-the-badge&logo=sourceforge&color=C45500&logoColor=FFD43B&labelColor=1B1818)](https://sourceforge.net/projects/warlock-studio/)

<br>

**Warlock-Studio** is a unified, high-performance platform for **upscaling, restoration, denoising, and frame interpolation**.
Inspired by [Djdefrag](https://github.com/Djdefrag) tools such as **QualityScaler** and **FluidFrames**.

---

## 📥 <span style="color:#FFD700;">Download Installer</span>

<div align="center">
  <p style="color:#ccc; font-size:14px; line-height: 1.6;">
    This installer was built using <b>PyInstaller</b> and <b>Inno Setup</b>.<br>
    By default, it includes <b>DirectML</b> support to ensure maximum compatibility with any graphics card (NVIDIA/AMD/INTEL).
  </p>
  <p style="color:#ccc; font-size:14px; margin-top: 15px;">
    Select your preferred option to download the latest version (Direct Release/SourceForge):
  </p>
</div>

<table style="width:100%; border-collapse:collapse;">
  <tr>
    </td>
    <td align="center" style="vertical-align:top; padding:10px;">
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v5.1.1/Warlock-Studio-5.1.1-Full-Installer.exe">
        <img src="rsc/GitHub_Logo_WS.png" alt="Download from GitHub"
             width="300" style="display:block; margin:auto; margin-bottom:10px;" />
      </a>
    <td align="center" style="vertical-align:top; padding:px;">
      <a href="https://sourceforge.net/projects/warlock-studio/" target="_blank">
        <img src="https://sourceforge.net/cdn/syndication/badge_img/3880091/oss-rising-star-black"
             alt="Warlock-Studio on SourceForge"
             width="190" style="display:block; margin:auto; margin-bottom:1px;" />
      </a>
    </td>
  </tr>
</table>

---

## 🖼️ Interface Previews

[Capture.webm](https://github.com/user-attachments/assets/86a679c7-ddf5-454f-9a05-db590cfced09)

![Capture](rsc/Capture2.png)
![Capture](rsc/Capture3.png)
![Capture](rsc/Capture4.png)
![Capture](rsc/Capture5.png)

---

## 🔍 Quality Comparison

[WsvideovsGit.webm](https://github.com/user-attachments/assets/c72f389d-827e-49b9-91b7-fd13e5b59f22)

[WsvideovsGit2.webm](https://github.com/user-attachments/assets/6695cce2-f42f-4955-8b43-56ec6d7b0bd2)

---

## ✨ Key Features

- **AI Upscaling & Restoration** – Utilize **Real-ESRGAN, BSRGAN, RealESRNet, RealESR_Animex4, and IRCNN** models for denoising, super-resolution, and detail recovery.
- **Face Restoration (GFPGAN)** – Recover facial details from low-resolution or blurry images and video frames.
- **Frame Interpolation (RIFE)** – Smooth motion or generate slow-motion content with **2×, 4×, or 8× interpolation**.
- **Advanced Hardware Acceleration** – Intelligent provider selection prioritizes **CUDA**, falls back to **DirectML**, and finally **CPU** for maximum compatibility.
- **Batch Processing** – Process multiple media files simultaneously, saving time and effort.
- **Custom Workflows** – Fine-grained control over models, resolution, output formats, and quality parameters.
- **Open-Source & Extensible** – Fully MIT licensed, for contributors and developers.

---

## 🖥️ System Requirements

| Component   | Minimum Specification | Recommended Specification              |
| :---------- | :-------------------- | :------------------------------------- |
| **OS**      | Windows 10 (64-bit)   | Windows 11 (64-bit)                    |
| **RAM**     | 8 GB                  | 16 GB+ (Recommended for 4K Video)      |
| GPU         | DirectX 12 Compatible | NVIDIA RTX 2060 / AMD RX 6700 XT       |
| **VRAM**    | 4 GB                  | 8 GB+ (NEO Engine auto-tunes limits)   |
| **Storage** | HDD Space             | NVMe SSD (Highly recommended for RIFE) |

---

## 🤝 Contributions

We welcome contributions from the community:

1. **Fork** the repository.
2. **Create a branch** for your feature or bug fix.
3. **Submit a Pull Request** with a detailed description and testing notes.

📧 Contact: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**

---

## 📜 License & Credits

© 2025 **Iván Eduardo Chavez Ayub**
<br>Licensed under **MIT**. Additional terms and attributions are provided in **NOTICE.md**.

### 📊 Integrated Technologies & Licenses

| Technology / Model              | License      | Author / Maintainer        | Source                                                     |
| :------------------------------ | :----------- | :------------------------- | :--------------------------------------------------------- |
| **Real-ESRGAN**                 | BSD 3-Clause | Xintao Wang                | [GitHub](https://github.com/xinntao/Real-ESRGAN)           |
| • RealESRGANx4                  | BSD 3-Clause | Xintao Wang                | Same as above                                              |
| • RealESRNetx4                  | BSD 3-Clause | Xintao Wang                | Same as above                                              |
| • RealESR_Gx4 (Custom Variant)  | BSD 3-Clause | Xintao / Community         | Same as above                                              |
| • RealESR_Animex4 (Anime Model) | BSD 3-Clause | Community                  | Same as above                                              |
| **BSRGAN**                      | Apache 2.0   | Kai Zhang                  | [GitHub](https://github.com/cszn/BSRGAN)                   |
| • BSRGANx4                      | Apache 2.0   | Kai Zhang                  | Same as above                                              |
| • BSRGANx2                      | Apache 2.0   | Kai Zhang                  | Same as above                                              |
| **IRCNN**                       | BSD / Mixed  | Kai Zhang                  | [GitHub](https://github.com/cszn/IRCNN)                    |
| • IRCNN_Mx1                     | BSD / Mixed  | Kai Zhang                  | Same as above                                              |
| • IRCNN_Lx1                     | BSD / Mixed  | Kai Zhang                  | Same as above                                              |
| **GFPGAN**                      | Apache 2.0   | TencentARC                 | [GitHub](https://github.com/TencentARC/GFPGAN)             |
| **RIFE**                        | Apache 2.0   | hzwer                      | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| **QualityScaler**               | MIT          | Djdefrag                   | [GitHub](https://github.com/Djdefrag/QualityScaler)        |
| **FluidFrames**                 | MIT          | Djdefrag                   | [GitHub](https://github.com/Djdefrag/FluidFrames)          |
| **ONNX Runtime**                | MIT          | Microsoft                  | [GitHub](https://github.com/microsoft/onnxruntime)         |
| **FFmpeg**                      | LGPL / GPL   | FFmpeg Team                | [Official Site](https://ffmpeg.org)                        |
| **ExifTool**                    | Artistic     | Phil Harvey                | [Official Site](https://exiftool.org/)                     |
| **Python**                      | PSF License  | Python Software Foundation | [Official Site](https://www.python.org)                    |
| **PyInstaller**                 | GPLv2+       | PyInstaller Team           | [GitHub](https://github.com/pyinstaller/pyinstaller)       |
| **Inno Setup**                  | Custom       | Jordan Russell             | [Official Site](http://www.jrsoftware.org/isinfo.php)      |


