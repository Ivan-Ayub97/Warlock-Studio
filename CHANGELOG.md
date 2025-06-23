# 📝 CHANGELOG — Warlock-Studio v2.1

**Release Date:** June 23, 2025

---

## 🚀 Major Enhancements & Stability Overhaul

This version focuses on massive improvements to stability, error handling, and code robustness, ensuring a smoother and more reliable user experience.

- 🛡️ **Robust Error Handling:**

  - Implemented comprehensive `try...except` blocks for AI model loading (`AI_upscale` & `AI_interpolation`) to prevent crashes if a model file is missing or corrupt.
  - Enhanced video frame extraction (`extract_video_frames`) with checks for file existence, successful video opening, and valid frame counts.
  - Made video encoding (`video_encoding`) more resilient by handling FFmpeg subprocess errors gracefully and providing clearer error messages.
  - Added a fallback for audio passthrough failures; the application now saves the video without audio instead of failing the entire process.

- 🧵 **Safe Thread & Process Management:**

  - Replaced the unsafe thread-stopping mechanism (which intentionally raised an error) with a modern, safe `threading.Event` (`stop_thread_flag`).
  - Ensures clean and predictable termination of background monitoring threads.

- ⚙️ **Resilient Core Processing:**
  - Added pre-flight checks to the metadata copy function (`copy_file_metadata`) to ensure `exiftool.exe` and source/destination files exist before execution.

---

## 🎨 UI/UX Refinements

- 🎨 **Refined Color Palette:**
  - Updated the main application theme for a new aesthetic.
  - App Name Color (`app_name_color`) changed to a golden yellow (`#ECD125`).
  - Widget Background Color (`widget_background_color`) changed to a deep red (`#960707`).
  - Default active button border color updated to red to match the new theme.

---

## 🔧 Code & Maintainability Improvements

- 🧹 **Improved Code Organization:**

  - Refactored file extension lists into clearer, separate categories: `supported_image_extensions` and `supported_video_extensions`.

- 📦 **Dependency and Initialization:**
  - Added new standard library imports (`shutil.move`, `subprocess.CalledProcessError`, `threading.Event`) to support the stability enhancements.
  - Ensured safer initialization of global variables at startup.

---

# 📝 **CHANGELOG — Warlock-Studio v2.0**

**Release Date:** June 6, 2025

---

## 🚀 Major Features

- 🧠 **AI Frame Interpolation Support (New Module):**

  - Introduced a new class `AI_interpolation`, providing frame generation capabilities using RIFE-based ONNX models.
  - Allows generation of 1 (x2), 3 (x4), or 7 (x8) intermediate frames, including slow-motion variants.
  - Enables temporal upscaling of video via AI.

- 🎥 **RIFE Models Integration:**
  - Added `RIFE` and `RIFE_Lite` to supported models.
  - Interpolation model list introduced: `RIFE_models_list`.
  - Extended `AI_models_list` to include all model types: SRVGGNetCompact, BSRGAN, IRCNN, and RIFE.

---

## ✨ Enhancements

- 🎨 **Visual/UI Redesign:**

  - App renamed to `"Warlock-Studio"` (with hyphen).
  - Background and widget colors updated: darker tones (`#121212`, `#454242`).
  - Text color changed to bright white (`#FFFFFF`).
  - App name highlighted in red (`#FF0E0E`).

- 📂 **Version-Specific User Preferences:**

  - Preferences now saved under versioned filenames like `Warlock-Studio_2.0_UserPreference.json`.
  - Prevents conflict with previous versions' configuration.

- 🧬 **Modular and Scalable Layout System:**

  - New GUI constants introduced (e.g. `offset_y_options`, `column_1_5`, `column_2_9`, etc.).
  - Enables more fine-tuned interface arrangement.

- 💾 **Extended File Type Compatibility:**

  - Added new image/video file extensions to `supported_file_extensions` and `supported_video_extensions`.
  - Ensures broader compatibility with input formats.

- 🚀 **Improved GPU Execution Support:**
  - Enhanced logic for selecting GPU via `DirectML`.
  - Supports up to 4 GPUs (`Auto`, `GPU 1` to `GPU 4`) via `provider_options`.

---

## 🔧 Technical Refinements

- 🧹 **Better Model List Structure:**

  - Models now grouped logically by type with `MENU_LIST_SEPARATOR`.
  - Makes UI dropdowns cleaner and more organized.

- 🧪 **Advanced Interpolation Logic:**

  - Support for dynamic multi-frame generation with tree-based logic (e.g. A-B-C from D).

- 📊 **Improved Numeric Precision and Postprocessing:**
  - Improved handling of floating-point range and normalization.
  - Enhanced logic for RGB/RGBA conversion and alpha blending.

---

## 🦖 UI/UX Refinements

- 📏 **Resizable Message Dialogs:**

  - MessageBox window can now be resized by the user (`resizable(True, True)`).

- 👌 **Improved Dialog Formatting:**
  - Better spacing and ordering of message elements.
  - Cleaner font use and default value display.

---

## ✅ Minor Fixes

- 🔧 Fixed UI typo: corrected `ttext_color` to `text_color`.
- 🪯 Enhanced inline comments and structural organization for better readability and maintainability.

---

# 📝 **CHANGELOG — Warlock-Studio v1.1**

**Release Date:** May 20, 2025

---

## ✨ Major Improvements

- ⚡ **Program Startup Optimization:**
  Startup time has been significantly reduced.

- 📦 **Model Loading Improvements:**
  Improved loading speed for models.

- 🔧 **General Performance Optimization:**
  Refactored core components. Background processing is now more efficient and uses fewer resources.

---

## 🐛 Minor Fixes

- Minor UI adjustments to improve accessibility.
