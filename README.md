![Warlock-Studio banner](Assets/banner.png)

<div align="center">

![Platform](https://img.shields.io/badge/Platform-Windows_10_%7C_11-003B6F?style=for-the-badge&logo=windows&logoColor=FFD43B&labelColor=1B1818)
![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=FFD43B&labelColor=1B1818)
![License: MIT](https://img.shields.io/badge/License-MIT-0A3B1E?style=for-the-badge&logo=open-source-initiative&logoColor=FFD43B&labelColor=1B1818)

[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&logo=git&color=6A1B9A&logoColor=FFD43B&labelColor=1B1818)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)
![Version 6](https://img.shields.io/badge/Version-6.0-660066?style=for-the-badge&logo=tag&logoColor=FFD43B&labelColor=1B1818)

[![Downloads Total](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&logo=github&color=2E2E2E&labelColor=1B1818&logoColor=FFD43B)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![SF Downloads](https://img.shields.io/sourceforge/dt/warlock-studio?style=for-the-badge&logo=sourceforge&color=C45500&logoColor=FFD43B&labelColor=1B1818)](https://sourceforge.net/projects/warlock-studio/)

<br>

**Warlock Studio** is a unified platform for **upscaling, restoring, denoising, and interpolating frames in videos and images.**
It is inspired by and based on [Djdefrag](https://github.com/Djdefrag) tools such as **QualityScaler** and **FluidFrames**.

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
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v6.0/Warlock-Studio-6.0x64.exe">
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

## 🖼️ Interface Capture

[Capture.webm](https://github.com/user-attachments/assets/86a679c7-ddf5-454f-9a05-db590cfced09)

![UICapture](rsc/Capture2.png)
![UICapture](rsc/Capture3.png)

---

## 🔍 Quality Comparison

[WsvideovsGit3.webm](https://github.com/user-attachments/assets/1326f86a-614e-40ae-b3a9-44ae15f648c6)

[WsvideovsGit.webm](https://github.com/user-attachments/assets/c72f389d-827e-49b9-91b7-fd13e5b59f22)

[WsvideovsGit2.webm](https://github.com/user-attachments/assets/6695cce2-f42f-4955-8b43-56ec6d7b0bd2)

---

## ✨ Key Features

- **AI Upscaling & Restoration** – Utilize **Real-ESRGAN, BSRGAN, RealESRNet, RealESR_Animex4, and IRCNN** models for denoising, super-resolution, and detail recovery.
- **Face Restoration (GFPGAN)** – Recover facial details from low-resolution or blurry images and video frames.
- **Frame Interpolation (RIFE)** – Smooth motion or generate slow-motion content with **2×, 4×, or 8× interpolation**.
- **Process Chaining** – Build sequential workflows by chaining steps. Mix **upscaling**, **face restoration**, and **interpolation**; each step’s output becomes the next step’s input automatically. Includes model auto-discovery, per-step GPU/codec settings, and smart validation (e.g., RIFE requires video).
- **Advanced Hardware Acceleration** – Intelligent provider selection prioritizes **CUDA**, falls back to **DirectML**, and finally **CPU** for maximum compatibility.
- **Batch Processing** – Process multiple media files simultaneously, saving time and effort.
- **Custom Workflows** – Fine-grained control over models, resolution, output formats, and quality parameters.
- **Open-Source & Extensible** – Fully MIT licensed, for contributors and developers.

---

## 🆕 New in v6.0 — Process Chaining

- Create multi-step pipelines; order steps to run sequentially per file.
- RIFE interpolation integrates as a chain step for video sources (graceful skip on images).
- Per-step model selection via a combobox fed by auto-discovered ONNX models in `AI-onnx/`.
- Automatic output routing: intermediate steps use temp folders; the final step writes to your chosen output path.
- Smart extension/codec correction by media type to prevent invalid outputs.
- Memory-safe execution with per-step VRAM tile sizing and cleanup between steps.

---

## 🖥️ System Requirements

<div align="center">

| <span style="color: #FBC02D;">Component</span> | <span style="color: #FBC02D;">Minimum Specification</span> | <span style="color: #FBC02D;">Recommended Specification</span> |
| :--- | :--- | :--- |
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **RAM** | 8 GB | 16 GB+ (Required for 4K & High-FPS Video) |
| **GPU** | DirectX 12 Compatible (DML) / NVIDIA GTX 10-Series | NVIDIA RTX 3060+ / AMD RX 6000+ |
| **VRAM** | 4 GB | 8 GB - 12 GB+ (For Stable Diffusion/Video Interpolation) |
| **Storage** | 2 GB available space | SSD (Critical for RIFE & Temp Video Processing) |
| **Architecture** | x64 | x64 (Native DirectML Support) |

</div>

> **Performance Tip:** Given that **Warlock Studio** leverages **DirectML** for hardware acceleration, keeping your GPU drivers updated is essential for maximizing processing speed across NVIDIA, AMD, and Intel hardware.

---

## 🤝 Contributions

We welcome contributions from the community.

📧 Contact: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**

---

## 📜 License & Credits

© 2025 **Iván Eduardo Chavez Ayub**
<br>Licensed under **MIT**. Additional terms and attributions are provided in **NOTICE.md**.

### 📊 Integrated Technologies & Licenses

| Technology / Model | License | Author / Maintainer | Source |
| :--- | :--- | :--- | :--- |
| **Real-ESRGAN** | BSD 3-Clause | Xintao Wang | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| • RealESRGANx4 | BSD 3-Clause | Xintao Wang | Same as above |
| • RealESRNetx4 | BSD 3-Clause | Xintao Wang | Same as above |
| • RealESR_Gx4 | BSD 3-Clause | Xintao / Community | Same as above |
| • RealESR_Animex4 | BSD 3-Clause | Community | Same as above |
| **BSRGAN** | Apache 2.0 | Kai Zhang | [GitHub](https://github.com/cszn/BSRGAN) |
| • BSRGANx4 | Apache 2.0 | Kai Zhang | Same as above |
| • BSRGANx2 | Apache 2.0 | Kai Zhang | Same as above |
| **IRCNN** | BSD / Mixed | Kai Zhang | [GitHub](https://github.com/cszn/IRCNN) |
| • IRCNN_Mx1 | BSD / Mixed | Kai Zhang | Same as above |
| • IRCNN_Lx1 | BSD / Mixed | Kai Zhang | Same as above |
| **GFPGAN** | Apache 2.0 | TencentARC | [GitHub](https://github.com/TencentARC/GFPGAN) |
| **RIFE** | MIT | Hzwer / Megvii | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| **QualityScaler** | MIT | Djdefrag | [GitHub](https://github.com/Djdefrag/QualityScaler) |
| **FluidFrames** | MIT | Djdefrag | [GitHub](https://github.com/Djdefrag/FluidFrames) |
| **DirectML** | MIT | Microsoft | [GitHub](https://github.com/microsoft/DirectML) |
| **ONNX Runtime** | MIT | Microsoft | [GitHub](https://github.com/microsoft/onnxruntime) |
| **CustomTkinter** | MIT | Tom Schimansky | [GitHub](https://github.com/TomSchimansky/CustomTkinter) |
| **TkinterDnD2** | MIT | pmgagne | [GitHub](https://github.com/pmgagne/tkinterdnd2) |
| **OpenCV (cv2)** | Apache 2.0 | OpenCV Team | [Official Site](https://opencv.org/) |
| **NumPy** | BSD 3-Clause | NumPy Developers | [Official Site](https://numpy.org/) |
| **Pillow (PIL)** | HPND | Python-Pillow Team | [GitHub](https://github.com/python-pillow/Pillow) |
| **MoviePy** | MIT | Zulko | [GitHub](https://github.com/Zulko/moviepy) |
| **FFmpeg** | LGPL / GPL | FFmpeg Team | [Official Site](https://ffmpeg.org) |
| **ExifTool** | Artistic | Phil Harvey | [Official Site](https://exiftool.org/) |
| **Psutil** | BSD 3-Clause | Giampaolo Rodola | [GitHub](https://github.com/giampaolo/psutil) |
| **WMI** | MIT | Tim Golden | [GitHub](https://github.com/tjguk/wmi) |
| **GPUtil** | MIT | Anders Krogh | [GitHub](https://github.com/anderskm/gputil) |
| **Requests** | Apache 2.0 | Kenneth Reitz | [GitHub](https://github.com/psf/requests) |
| **Packaging** | Apache 2.0 | PyPA | [GitHub](https://github.com/pypa/packaging) |
| **Natsort** | MIT | Seth M. Morton | [GitHub](https://github.com/SethMMorton/natsort) |
| **Python** | PSF License | Python Software Foundation | [Official Site](https://www.python.org) |
| **PyInstaller** | GPLv2+ | PyInstaller Team | [GitHub](https://github.com/pyinstaller/pyinstaller) |
| **Inno Setup** | Custom | Jordan Russell | [Official Site](http://www.jrsoftware.org/isinfo.php) |
