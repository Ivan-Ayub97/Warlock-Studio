![Warlock-Studio banner](Assets/banner.png)

<div align="center">

![Platform](https://img.shields.io/badge/Platform-Windows_10_%7C_11-003B6F?style=for-the-badge&logo=windows&logoColor=FFD43B&labelColor=1B1818)
![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?style=for-the-badge&logo=python&logoColor=FFD43B&labelColor=1B1818)
![License: MIT](https://img.shields.io/badge/License-MIT-0A3B1E?style=for-the-badge&logo=open-source-initiative&logoColor=FFD43B&labelColor=1B1818)

[![Last Commit](https://img.shields.io/github/last-commit/Ivan-Ayub97/Warlock-Studio?style=for-the-badge&logo=git&color=6A1B9A&logoColor=FFD43B&labelColor=1B1818)](https://github.com/Ivan-Ayub97/Warlock-Studio/commits/main)

[![Downloads Total](https://img.shields.io/github/downloads/Ivan-Ayub97/Warlock-Studio/total?style=for-the-badge&logo=github&color=2E2E2E&labelColor=1B1818&logoColor=FFD43B)](https://github.com/Ivan-Ayub97/Warlock-Studio/releases)
[![SF Downloads](https://img.shields.io/sourceforge/dt/warlock-studio?style=for-the-badge&logo=sourceforge&color=C45500&logoColor=FFD43B&labelColor=1B1818)](https://sourceforge.net/projects/warlock-studio/)



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

* **AI Upscaling & Restoration** – Utilize **Real-ESRGAN, BSRGAN, RealESRNet, RealESR_Animex4, and IRCNN** models for denoising, super-resolution, and detail recovery.
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
| GPU | DirectX 12 Compatible | NVIDIA RTX 2060 / AMD RX 6700 XT | NVIDIA RTX 4070 Ti / AMD RX 7900 XT or better |
| **VRAM** | 4 GB | 8 GB+ (NEO Engine auto-tunes limits) |
| **Storage** | HDD Space | NVMe SSD (Highly recommended for RIFE) |

---

## 🚀 What's Next?

We are working on a **fundamental transformation** of how users create, manage, and execute pipelines within **Warlock-Studio**. The next major update introduces the **Visual Nodes** system, a powerful tool that will drastically improve the user experience and the complexity of the projects you can handle.

### 🖼️ New Module: `WSNodes` (Visual Node Editor)

The **`WSNodes`** module is an advanced visual node editor, built on **PyQt5**, that will allow users to:

* **Design Pipelines Intuitively:** Instead of directly editing JSON files, you will be able to drag, connect, and configure modules as if you were drawing a flowchart.
* **Simplified Visual Complexity:** Build complex, multi-step, and branching workflows in a clear and visual manner.
* **Reusability and Modularity:** Each node will encapsulate a specific function, allowing for easy reuse and simpler code maintenance.
* **Compilation and Execution:** Once designed, the editor will automatically compile the visual diagram into the internal pipeline format that **Warlock-Studio** already uses for execution.

**This represents a significant leap in Warlock-Studio's usability and capability, moving it towards a more flexible and visually powerful project development environment.**

![WSNodes](rsc/Capture_Pipeline.png)
![WSNodes](rsc/WSNodes.png)

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

| Technology / Model | License | Author / Maintainer | Source |
| :--- | :--- | :--- | :--- |
| **Real-ESRGAN** | BSD 3-Clause | Xintao Wang | [GitHub](https://github.com/xinntao/Real-ESRGAN) |
| • RealESRGANx4 | BSD 3-Clause | Xintao Wang | Same as above |
| • RealESRNetx4 | BSD 3-Clause | Xintao Wang | Same as above |
| • RealESR_Gx4 (Custom Variant) | BSD 3-Clause | Xintao / Community | Same as above |
| • RealESR_Animex4 (Anime Model) | BSD 3-Clause | Community | Same as above |
| **BSRGAN** | Apache 2.0 | Kai Zhang | [GitHub](https://github.com/cszn/BSRGAN) |
| • BSRGANx4 | Apache 2.0 | Kai Zhang | Same as above |
| • BSRGANx2 | Apache 2.0 | Kai Zhang | Same as above |
| **IRCNN** | BSD / Mixed | Kai Zhang | [GitHub](https://github.com/cszn/IRCNN) |
| • IRCNN_Mx1 | BSD / Mixed | Kai Zhang | Same as above |
| • IRCNN_Lx1 | BSD / Mixed | Kai Zhang | Same as above |
| **GFPGAN** | Apache 2.0 | TencentARC | [GitHub](https://github.com/TencentARC/GFPGAN) |
| **RIFE** | Apache 2.0 | hzwer | [GitHub](https://github.com/megvii-research/ECCV2022-RIFE) |
| **QualityScaler** | MIT | Djdefrag | [GitHub](https://github.com/Djdefrag/QualityScaler) |
| **FluidFrames** | MIT | Djdefrag | [GitHub](https://github.com/Djdefrag/FluidFrames) |
| **ONNX Runtime** | MIT | Microsoft | [GitHub](https://github.com/microsoft/onnxruntime) |
| **FFmpeg** | LGPL / GPL | FFmpeg Team | [Official Site](https://ffmpeg.org) |
| **ExifTool** | Artistic | Phil Harvey | [Official Site](https://exiftool.org/) |
| **Python** | PSF License | Python Software Foundation | [Official Site](https://www.python.org) |
| **PyInstaller** | GPLv2+ | PyInstaller Team | [GitHub](https://github.com/pyinstaller/pyinstaller) |
| **Inno Setup** | Custom | Jordan Russell | [Official Site](http://www.jrsoftware.org/isinfo.php) |














