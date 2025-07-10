## Version 2.2

**Release date:** 7 July 2025

### 1. Major Enhancements and Stability Overhaul

1.1 **Comprehensive Logging System**

- Implemented a full-featured `logging` system that writes to files in the user's `Documents` folder (`warlock_studio.log` and `error_log.txt`). This provides detailed diagnostics for debugging without relying solely on console output.
- A unified `log_and_report_error` function centralizes error handling, ensuring all critical issues are both logged and displayed to the user.

  1.2 **Proactive Environment Validation**

- The application now performs pre-flight checks before processing begins to prevent common failures.
- Includes validation for Python version, required modules (`validate_environment`), FFmpeg availability, disk space, and available RAM (`validate_system_requirements`).
- Verifies that all input file paths exist and are accessible (`validate_file_paths`) and that the output directory is writable (`validate_output_path`).

  1.3 **Resilient Video Encoding Pipeline**

- The entire `video_encoding` function was overhauled for maximum reliability.
- **Codec Fallback:** The system now tests for hardware codec availability (NVENC, AMF, QSV) before encoding. If a selected hardware encoder is not functional, it automatically falls back to the highly compatible `libx264` software encoder.
- **Robust Audio Handling:** Implements a fallback chain for audio processing. It first attempts to directly copy the audio stream; if that fails, it attempts to re-encode it; if that also fails, it finalizes the video without audio, ensuring a video file is always produced.

  1.4 **Graceful Shutdown and Cleanup**

- Implemented `atexit` and `signal` handlers to ensure that temporary files are cleaned up and child processes are terminated safely, even on unexpected exits.
- Replaced abrupt `process.kill()` calls with the more graceful `process.terminate()` to allow for cleaner process shutdown.

### 2. Performance and Memory Optimization

2.1 **Aggressive Memory Management**

- Video frame processing (`upscale_video_frames_async`) no longer holds large numbers of frames in RAM. It now writes small batches to disk and immediately calls the garbage collector (`gc.collect()`) to free memory, dramatically reducing the risk of crashes on long videos.

  2.2 **Dynamic GPU VRAM Error Recovery**

- The AI orchestration logic can now recover from GPU "out of memory" errors during tiling. If an error is detected, it automatically reduces the tile resolution and retries the operation on that specific frame, preventing a total process failure.

### 3. Critical Bug Fixes

3.1 **Resolved Video Encoding Race Condition**

- Fixed a critical bug where video encoding could start before all frame-writing threads were complete. The system now tracks all writer threads and explicitly waits for them to finish (`thread.join()`) before beginning the final video encoding, preventing corrupted or incomplete videos.

  3.2 **Corrected Persistent Stop Flag**

- The `stop_thread_flag` is now reset (`.clear()`) at the start of each "Make Magic" execution, fixing a bug where a previously stopped job would prevent a new one from running.

  3.3 **Eliminated Status Update Race Condition**

- Implemented a `threading.Lock` (`global_status_lock`) to protect shared flags that update the GUI. This prevents race conditions where multiple threads could attempt to modify the status simultaneously.

### 4. UI / UX Refinements

4.1 **Updated Splash Screen**

- Reduced splash screen duration to 10 seconds for a faster application start-up.
- Corrected asset path to `Assets/banner.png` for proper display.

  4.2 **New Application Theme**

| Element           | New Value        | Old Value (v2.1)     |
| :---------------- | :--------------- | :------------------- |
| App name          | `#FF0000` (Red)  | `#ECD125` (Gold)     |
| Widget background | `#5A5A5A` (Grey) | `#960707` (Dark Red) |
| Accent/Border     | Gold & Red       | Blue & Red           |

### 5. Codebase Health and Maintainability

5.1 **Enhanced Checkpointing and Recovery**

- Added functions (`create_checkpoint`, `load_checkpoint`) to save and resume the progress of video frame processing, allowing recovery from interruptions.

  5.2 **Hardened Core Methods**

- Core methods in AI classes now include checks for `None` inputs and feature default fallbacks (`case _:`) in `match` statements to prevent unexpected errors with unsupported data.

## Version 2.1

**Release date:** 23 June 2025

### 1. Major Enhancements and Stability Overhaul

1.1 **Robust Error Handling**

- Model loading (`AI_upscale`, `AI_interpolation`) wrapped in `try…except FileNotFoundError, OSError`; meaningful error messages propagate to GUI.
- `extract_video_frames()` validates file existence, `cv2.VideoCapture.isOpened()`, and frame count > 0.
- `video_encoding()` captures `subprocess.CalledProcessError`, logs `stderr`, and continues with fallback strategies.
- Audio passthrough failures now trigger a silent audio‑less encode instead of total job abort.

  1.2 **Safe Thread and Process Management**

- Deprecated error‑raising thread stop replaced with `threading.Event` (`stop_thread_flag`) polled at defined checkpoints.

  1.3 **Resilient Core Processing**

- `copy_file_metadata()` now verifies `exiftool.exe` availability and the existence of source/target before execution.

### 2. UI / UX Refinements

2.1 **Refined Colour Palette**

| Element           | New Value                       |
| ----------------- | ------------------------------- |
| App name          | `#ECD125`                       |
| Widget background | `#960707`                       |
| Active border     | Red (same as widget background) |

### 3. Code‑base Maintainability

3.1 **Improved Code Organisation**

- File‑extension lists extracted to `filetypes.py` as `SUPPORTED_IMAGE_EXTENSIONS`, `SUPPORTED_VIDEO_EXTENSIONS`.

  3.2 **Dependency and Initialisation**

- Added imports: `shutil.move`, `subprocess.CalledProcessError`, `threading.Event`.
- Global variables initialised in `init_globals()` for deterministic start‑up.

---

## Version 2.0

**Release date:** 6 June 2025

### 1. Major Features

1.1 **AI Frame Interpolation Support** (`AI_interpolation` class)

- Supports RIFE‑based ONNX models; generates 1 (×2), 3 (×4), or 7 (×8) intermediate frames.
- Provides both real‑time preview and batch processing modes.
- Integrates with `FrameScheduler` for temporal upscaling pipelines.

  1.2 **RIFE Models Integration**

- Added **RIFE** and **RIFE_Lite** to model repository.
- `RIFE_models_list` enumerates available checkpoints; `AI_models_list` now merges SRVGGNetCompact, BSRGAN, IRCNN, and RIFE families.

### 2. Enhancements

2.1 **Visual/UI Redesign**

- Application renamed to **“Warlock‑Studio”** (with hyphen).
- New dark palette (`#121212`, `#454242`) with bright white text (`#FFFFFF`) and accent red (`#FF0E0E`).

  2.2 **Version‑Specific User Preferences**

- User configuration stored as `Warlock-Studio_<major>.<minor>_UserPreference.json` to avoid backward‑compatibility clashes.

  2.3 **Modular and Scalable Layout System**

- Added GUI constants defined in `layout_constants.py` (e.g., `OFFSET_Y_OPTIONS`, `COLUMN_1_5`).

  2.4 **Extended File‑Type Compatibility**

- Updated `SUPPORTED_FILE_EXTENSIONS` and `SUPPORTED_VIDEO_EXTENSIONS` to include modern codecs (e.g., HEIC, AVIF, WebM).

  2.5 **Improved GPU Execution Support**

- `provider_options` enumerates up to four DirectML devices (Auto, GPU 1 – GPU 4); selection persists across sessions.

### 3. Technical Refinements

3.1 **Model List Structure**

- Menu drop‑downs now grouped by category separated by `MENU_LIST_SEPARATOR` for readability.

  3.2 **Advanced Interpolation Logic**

- Implements tree‑based frame generation (e.g., D→A‑B‑C) with dependency tracking to avoid redundant inference passes.

  3.3 **Improved Numeric Precision and Post‑Processing**

- Normalisation uses 32‑bit floats with epsilon guarding; RGBA conversion paths optimised using `numexpr`.

### 4. UI / UX Refinements

4.1 **Resizable Message Dialogs** (`MessageBox`; Tk `resizable(True, True)`).
4.2 **Improved Dialog Formatting** – uniform spacing, font hierarchy, and default‑value display.

### 5. Minor Fixes

- Corrected typo `ttext_color` → `text_color`.
- Expanded inline comments and reorganised sections for clarity.

---

## Version 1.1

**Release date:** 20 May 2025

### 1. Major Improvements

1.1 **Program Start‑up Optimisation** – launch time reduced via lazy module loading.
1.2 **Model Loading Improvements** – parallel prefetch and checksum verification.
1.3 **General Performance Optimisation** – core refactor, improved I/O scheduling, and smarter resource allocation.

### 2. Minor Fixes

- User‑interface tweaks for better accessibility (focus indicators, tab order).
