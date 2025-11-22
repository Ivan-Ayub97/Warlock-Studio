![Warlock-Studio banner](Assets/banner.png)

<div align="center">

[![Version](https://img.shields.io/badge/Version-4.3-FF4500?style=for-the-badge&logo=git&logoColor=white)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases/tag/4.3)
[![License](https://img.shields.io/badge/License-MIT-6A0DAD?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&color=2E8B57&logo=git&logoColor=white)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)

[![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?style=for-the-badge&logo=windows11&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=FFD700)](https://www.python.org/)
[![Downloads](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total.svg?style=for-the-badge&color=FFD700&logo=download&logoColor=black)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)

Inspired by [Djdefrag](https://github.com/Djdefrag) tools such as **QualityScaler** and **FluidFrames**, **Warlock-Studio** provides a unified, high-performance platform for **upscaling, restoration, denoising, and frame interpolation**.

---

## 📥 <span style="color:#00ffcc;">Download Installer</span>

<div style="color:#ccc; font-size:14px; margin-top:-6px;">
  You can download the installer from either option below:
</div>

<table style="width:100%; border-collapse:collapse;">
  <tr>
    <td align="center" style="vertical-align:top; padding:px;">
      <a href="https://sourceforge.net/projects/warlock-studio/" target="_blank">
        <img src="https://sourceforge.net/cdn/syndication/badge_img/3880091/oss-rising-star-black"
             alt="Warlock-Studio on SourceForge"
             width="190" style="display:block; margin:auto; margin-bottom:1px;" />
      </a>
    </td>
    <td align="center" style="vertical-align:top; padding:10px;">
      <a href="https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/v5.0/Warlock-Studio-5.0-Full-Installer.exe" target="_blank">
        <img src="rsc/GitHub_Logo_WS.png" alt="Download from GitHub"
             width="300" style="display:block; margin:auto; margin-bottom:10px;" />
      </a>
    </td>
  </tr>
</table>

---

## 🖼️ Interface Previews

![interface](rsc/Capture.png)

---

## 🖼️ Quality Comparison

[WsvideovsGit.webm](https://github.com/user-attachments/assets/c72f389d-827e-49b9-91b7-fd13e5b59f22)

[WsvideovsGit2.webm](https://github.com/user-attachments/assets/6695cce2-f42f-4955-8b43-56ec6d7b0bd2)

![Comparison](rsc/image_comparison.png)

---

## ✨ Update 5.0

- **Modular Architecture (v5.0):** The code has been refactored into specialized components (`core`, `preferences`, `console`, `drag_drop`) for **superior stability, fault isolation, and code maintainability**.
- **NEO Engine:** Advanced diagnostic subsystem that scans **VRAM, RAM, and CPU topology** to generate **optimal configuration recommendations** in real-time.
- **⚙️ Centralized Preferences Panel:** Provides a quick-access **Settings Button** that opens a comprehensive panel for system management:
  - **UI Control:** Customize the Theme (Light/Dark), Interface Scaling, and Window Opacity.
  - **Hardware Configuration:** Adjust the **VRAM limit**, select the preferred GPU device, and configure \texttt{tiling} parameters based on **NEO Engine** data.
  - **Maintenance:** Tools to clear the model cache, manage error logs, and perform online update checks.
- **Integrated Console:** A GUI terminal that intercepts `sys.stdout` and `sys.stderr` to display inference logs, errors, and warnings with **syntax highlighting** for instant debugging.
- **Native Drag & Drop:** Direct support for dragging and dropping image and video files from the operating system explorer.
- ***

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

- **OS:** Windows 11 or Windows 10 (64-bit)
- **RAM:** 8GB (minimum) / 16GB+ (recommended for 4K Video)
- **GPU:** DirectX 12 compatible graphics card. NVIDIA (for CUDA), AMD, or Intel GPU with up-to-date drivers.
- **VRAM:** 4GB+ recommended. The **NEO Engine** automatically tunes limits on startup.
- **Storage:** Sufficient free space for input and processed media. SSD highly recommended for video I/O.

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
Licensed under **MIT**. Additional terms and attributions are provided in **NOTICE.md**.

### 📊 Integrated Technologies & Licenses

| Technology     | License                | Author / Maintainer                      | Source                                                      |
| -------------- | ---------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| QualityScaler  | MIT                    | [Djdefrag](https://github.com/Djdefrag)  | [GitHub](https://github.com/Djdefrag/QualityScaler)         |
| FluidFrames    | MIT                    | [Djdefrag](https://github.com/Djdefrag)  | [GitHub](https://github.com/Djdefrag/FluidFrames)           |
| Real-ESRGAN    | BSD 3-Clause / Apache  | Xintao Wang                              | [GitHub](https://github.com/xinntao/Real-ESRGAN)            |
| GFPGAN         | Apache 2.0             | TencentARC / Xintao Wang                 | [GitHub](https://github.com/TencentARC/GFPGAN)              |
| RIFE           | Apache 2.0             | hzwer                                    | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE)  |
| BSRGAN         | Apache 2.0             | Kai Zhang                                | [GitHub](https://github.com/cszn/BSRGAN)                    |
| IRCNN          | BSD / Mixed            | Kai Zhang                                | [GitHub](https://github.com/cszn/IRCNN)                     |
| ONNX Runtime   | MIT                    | Microsoft                                | [GitHub](https://github.com/microsoft/onnxruntime)          |
| FFmpeg         | LGPL / GPL             | FFmpeg Team                              | [Official Site](https://ffmpeg.org)                         |
| ExifTool       | Artistic License       | Phil Harvey                              | [Official Site](https://exiftool.org/)                      |
| Python         | PSF License            | Python Software Foundation               | [Official Site](https://www.python.org)                     |
| PyInstaller    | GPLv2+                 | PyInstaller Team                         | [GitHub](https://github.com/pyinstaller/pyinstaller)        |
| Inno Setup     | Custom                 | Jordan Russell                           | [Official Site](http://www.jrsoftware.org/isinfo.php)       |

</div>
