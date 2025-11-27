![Warlock-Studio banner](Assets/banner.png)

<div align="center">

[![Platform](https://img.shields.io/badge/Platform-Windows_10_%7C_11-0A1E5A?style=for-the-badge&logo=windows&logoColor=0A1E5A&labelColor=1a1a1a)](https://microsoft.com)
[![Python](https://img.shields.io/badge/Python-3.10+-7B4F00?style=for-the-badge&logo=python&logoColor=7B4F00&labelColor=1a1a1a)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-8C6239?style=for-the-badge&logo=open-source-initiative&logoColor=8C6239&labelColor=1a1a1a)](LICENSE)

[![Downloads Total](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&logo=github&color=7B0F12&logoColor=7B0F12&labelColor=1a1a1a)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![SF Downloads](https://img.shields.io/sourceforge/dt/warlock-studio?style=for-the-badge&logo=sourceforge&color=A14F03&logoColor=A14F03&labelColor=1a1a1a)](https://sourceforge.net/projects/warlock-studio/)

[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=5A5A5A&logo=git&logoColor=5A5A5A&labelColor=1a1a1a)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)
[![Issues](https://img.shields.io/github/issues/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=6A0DAD&logo=github&logoColor=6A0DAD&labelColor=1a1a1a)](https://github.com/Ivan-Ayub97/Warlock-Studio/issues)
[![Issues Closed](https://img.shields.io/github/issues-closed/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=0F4D2C&logo=github&logoColor=0F4D2C&labelColor=1a1a1a)](https://github.com/Ivan-Ayub97/Warlock-Studio/issues?q=is%3Aissue+is%3Aclosed)
[![Commits Month](https://img.shields.io/github/commit-activity/m/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=7A1FA2&logo=github&logoColor=7A1FA2&labelColor=1a1a1a)](https://github.com/Ivan-Ayub97/Warlock-Studio)


<br>

**Warlock-Studio** is a unified, high-performance platform for **upscaling, restoration, denoising, and frame interpolation**.
<br>
Inspired by [Djdefrag](https://github.com/Djdefrag) tools such as **QualityScaler** and **FluidFrames**.

</div>

---

## 📥 <span style="color:#FFD700;">Download Installer</span>

<div align="center">
  <p style="color:#ccc; font-size:14px; line-height: 1.6;">
    This installer was built using <b>PyInstaller</b> and <b>Inno Setup</b>.<br>
    By default, it includes <b>DirectML</b> support to ensure maximum compatibility with any graphics card (NVIDIA/AMD/INTEL).
  </p>
  <p style="color:#ccc; font-size:14px; margin-top: 15px;">
    Select your preferred option to download the latest version (Direct Release/Mirror):
  </p>
</div>

<table align="center" style="width:100%; border-collapse:collapse; border:none;">
  <tr>
    </td>
    <td align="center" style="vertical-align:top; width:50%; border:none; padding:20px;">
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v5.0/Warlock-Studio-5.0-Full-Installer.exe" target="_blank">
        <img src="rsc/GitHub_Logo_WS.png" alt="Download from GitHub" width="300" style="display:block; margin:auto; margin-bottom:15px;" />
      </a>
    </td>
    <td align="center" style="vertical-align:top; width:50%; border:none; padding:20px;">
      <a href="https://sourceforge.net/projects/warlock-studio/" target="_blank">
        <img src="https://sourceforge.net/cdn/syndication/badge_img/3880091/oss-rising-star-black" alt="Warlock-Studio on SourceForge" width="200" style="display:block; margin:auto; margin-bottom:15px; border-radius:5px;" />
      </a>
  </tr>
</table>

---

## ✨ What is New in v5.0

The **v5.0** release represents a foundational transformation of the application:

* **🧩 Modular Architecture:** The core has been re-engineered into specialized components (`core`, `preferences`, `console`, `drag_drop`) for superior stability and fault isolation.
* **🧠 NEO Engine:** A new diagnostic subsystem that scans CPU topology, RAM, and GPU VRAM to automatically recommend optimal tiling and thread settings.
* **🖥️ Integrated Debug Console:** A real-time GUI terminal that intercepts `stdout` and `stderr` with syntax highlighting, allowing users to diagnose FFmpeg or ONNX issues instantly.
* **⚡ Native CUDA & Failover:** The backend now strictly prioritizes **CUDA** (NVIDIA Optimized) > **DirectML** > **CPU**, with corrected integer typing for device IDs.
* **💎 Lossless Pipeline:** Deprecated `.jpg` usage in favor of `.png` for intermediate frames to prevent compression artifacts (blurriness).

---

## 🖼️ Interface Previews

<div align="center">
  <img src="rsc/Capture.png" alt="Main Interface" style="border-radius: 10px; box-shadow: 0px 0px 20px rgba(0,0,0,0.5);">
</div>

<br>

### ⚙️ Preferences
[Preferences.webm](https://github.com/user-attachments/assets/933003de-7618-4ed4-8815-077c69bf1ebc)

---

## 🔍 Quality Comparison

[WsvideovsGit.webm](https://github.com/user-attachments/assets/c72f389d-827e-49b9-91b7-fd13e5b59f22)

[WsvideovsGit2.webm](https://github.com/user-attachments/assets/6695cce2-f42f-4955-8b43-56ec6d7b0bd2)

![Comparison](rsc/image_comparison.png)

---

## ✨ Key Features

* **AI Upscaling & Restoration** – Utilize **Real-ESRGAN, BSRGAN, and IRCNN** models for denoising, super-resolution, and detail recovery.
* **Face Restoration (GFPGAN)** – Recover facial details from low-resolution or blurry images and video frames.
* **Frame Interpolation (RIFE)** – Smooth motion or generate slow-motion content with **2×, 4×, or 8× interpolation**.
* **Advanced Hardware Acceleration** – Intelligent provider selection prioritizes **CUDA**, falls back to **DirectML**, and finally **CPU** for maximum compatibility.
* **Batch Processing** – Process multiple media files simultaneously, saving time and effort.
* **Custom Workflows** – Fine-grained control over models, resolution, output formats, and quality parameters.
* **Open-Source & Extensible** – Fully MIT licensed, for contributors and developers.

---

## 🖥️ System Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **RAM** | 8 GB | 16 GB+ (Recommended for 4K Video) |
| GPU | DirectX 12 Compatible | NVIDIA RTX 3060 / AMD RX 6700 XT | NVIDIA RTX 4070 Ti / AMD RX 7900 XT or better |
| **VRAM** | 4 GB | 8 GB+ (NEO Engine auto-tunes limits) |
| **Storage** | HDD Space | NVMe SSD (Highly recommended for RIFE) |

---

## 🤝 Contributions

We welcome contributions from the community:

1.  **Fork** the repository.
2.  **Create a branch** for your feature or bug fix.
3.  **Submit a Pull Request** with a detailed description and testing notes.

📧 Contact: **[negroayub97@gmail.com](mailto:negroayub97@gmail.com)**

---

## 📜 License & Credits

© 2025 **Iván Eduardo Chavez Ayub**
<br>Licensed under **MIT**. Additional terms and attributions are provided in **NOTICE.md**.

### 📊 Integrated Technologies & Licenses

| Technology | License | Author / Maintainer | Source |
| :--- | :--- | :--- | :--- |
| **QualityScaler** | MIT | [Djdefrag](https://github.com/Djdefrag) | [GitHub](https://github.com/Djdefrag/QualityScaler) |
| **FluidFrames** | MIT | [Djdefrag](https://github.com/Djdefrag) | [GitHub](https://github.com/Djdefrag/FluidFrames) |
| **Real-ESRGAN** | BSD 3-Clause | Xintao Wang | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| **GFPGAN** | Apache 2.0 | TencentARC | [GitHub](https://github.com/TencentARC/GFPGAN) |
| **RIFE** | Apache 2.0 | hzwer | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| **BSRGAN** | Apache 2.0 | Kai Zhang | [GitHub](https://github.com/cszn/BSRGAN) |
| **IRCNN** | BSD / Mixed | Kai Zhang | [GitHub](https://github.com/cszn/IRCNN) |
| **ONNX Runtime** | MIT | Microsoft | [GitHub](https://github.com/microsoft/onnxruntime) |
| **FFmpeg** | LGPL / GPL | FFmpeg Team | [Official Site](https://ffmpeg.org) |
| **ExifTool** | Artistic | Phil Harvey | [Official Site](https://exiftool.org/) |
| **Python** | PSF License | Python Software Foundation | [Official Site](https://www.python.org) |
| **PyInstaller** | GPLv2+ | PyInstaller Team | [GitHub](https://github.com/pyinstaller/pyinstaller) |
| **Inno Setup** | Custom | Jordan Russell | [Official Site](http://www.jrsoftware.org/isinfo.php) |






