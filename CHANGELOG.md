## Version 4.0.1

**Release date:** 27 July 2025

### Model Cleanup and Optimization

#### 1.1 **SuperResolution-10 Model Removal**

- **Model Deprecation**: Removed the SuperResolution-10 model from the application due to performance and compatibility issues.
- **Code Cleanup**: Eliminated the dedicated `AI_super_resolution` class and all related processing pipelines.
- **UI Updates**: Removed SuperResolution-10 from model selection dropdown and information dialogs.
- **Memory Optimization**: Cleaned up VRAM usage configurations by removing SuperResolution-10 entries (0.8 GB allocation).
- **Streamlined Processing**: Simplified the upscaling orchestrator by removing SuperResolution-specific routing logic.
- **Model List Cleanup**: Removed `SuperResolution_models_list` from the main AI models collection.

#### 1.2 **Performance Improvements**

- **Reduced Memory Footprint**: Application now uses less memory without the SuperResolution-10 model overhead.
- **Simplified Code Paths**: Cleaner processing logic with fewer conditional branches for model selection.
- **Enhanced Stability**: Removed potential failure points associated with the deprecated model.

---

## Version 4.0

**Release date:** 18 July 2025

### 1. AI Model Integration & Enhancement

#### 1.1 **SuperResolution-10 Model Implementation**

- **New AI Model Integration**: Added support for the SuperResolution-10 model, providing advanced super-resolution capabilities with 10x upscaling factor. This model is specifically designed for very low-resolution images and excels at significant resolution increases.
- **Specialized Processing Pipeline**: Created a dedicated `AI_super_resolution` class that inherits from the new `AI_model_base` class, implementing proper preprocessing (CHW format conversion, normalization) and postprocessing (HWC format conversion, value clipping) for optimal results.
- **Model Information Integration**: Added comprehensive model information in the AI model selector dialog, including year (2023), function (high-resolution image enhancement), and specialized use cases.

#### 1.2 **AI Architecture Improvements**

- **Base Class Implementation**: Created the missing `AI_model_base` class that provides common functionality for all AI models, including ONNX model loading with GPU acceleration support and proper error handling.
- **Enhanced Model Loading**: Implemented robust model loading with provider selection (DML, CPU) and comprehensive error handling for missing model files.
- **VRAM Management**: Added VRAM usage information for SuperResolution-10 (0.8 GB) to help users optimize their GPU memory usage.

#### 2. Code Quality & Stability

##### 2.1 **Import System Optimization**

- **Fixed Import Errors**: Resolved critical `NameError: name 'numpy_ndarray' is not defined` by properly organizing imports at the top of the file.
- **Consolidated Imports**: Removed duplicate import sections and properly structured the import hierarchy for better maintainability.
- **Type Annotation Fixes**: Corrected type annotations throughout the codebase to use the proper imported numpy types.

##### 2.2 **Enhanced Error Handling**

- **Model Loading Resilience**: Implemented try-catch blocks for model loading operations with meaningful error messages.
- **Graceful Degradation**: Added fallback mechanisms that return the original image if super-resolution enhancement fails, ensuring the application never crashes.
- **Debug Information**: Enhanced logging with model loading status and error reporting for better troubleshooting.

#### 3. User Interface Updates

##### 3.1 **Model Selection Enhancement**

- **Updated Model List**: SuperResolution-10 is now properly integrated into the AI model dropdown menu and categorized appropriately.
- **Information Dialog Updates**: Added detailed information about the SuperResolution-10 model in the help dialog, including its capabilities and recommended use cases.
- **Model Orchestration**: Enhanced the upscaling orchestrator to properly detect and route SuperResolution model tasks to the appropriate processing pipeline.

#### 4. Technical Improvements

##### 4.1 **Processing Pipeline Optimization**

- **Specialized Image Processing**: Implemented dedicated image processing functions for super-resolution models that handle the unique requirements of the SuperResolution-10 model.
- **Memory Efficiency**: Optimized image preprocessing and postprocessing to minimize memory usage during super-resolution operations.
- **Performance Monitoring**: Added processing time tracking for super-resolution operations to help users understand processing performance.

#### 4.2 **Integration Completeness**

- **Full Model Integration**: SuperResolution-10 is now fully integrated into all aspects of the application, from model selection to processing to output generation.
- **Consistent User Experience**: The super-resolution workflow follows the same patterns as other AI models, ensuring a consistent user experience.
- **Quality Assurance**: Implemented comprehensive testing to ensure the SuperResolution-10 model works correctly with both individual images and batch processing.

### 5. **Smart AI Model Distribution System**

#### 5.1 **Automatic Model Download**

- **Lightweight Installer**: Significantly reduced installer size from 1.4GB to approximately 300MB by removing AI models from the installation package.
- **On-Demand Download**: Implemented intelligent model downloading system that automatically fetches required AI models (327MB) when the application is first launched.
- **Progress Tracking**: Added visual progress indicators with download speed and completion percentage during model acquisition.
- **Fallback URLs**: Integrated multiple download sources (GitHub Releases, SourceForge) to ensure reliable model availability.
- **Resume Capability**: Download system supports resuming interrupted downloads and validates file integrity.

#### 5.2 **PyInstaller Optimization**

- **Optimized Packaging**: Updated `.spec` file to exclude AI model directory from executable packaging, reducing final executable size by over 1GB.
- **Enhanced Dependencies**: Added model downloader module to the build process with proper hidden imports for requests, threading, and file handling libraries.
- **Improved Compression**: Increased optimization level and added module exclusions to further reduce executable size.

#### 5.3 **Installation Experience**

- **Smart Setup Script**: Created enhanced Inno Setup configuration that can optionally download models during installation or defer to first-run.
- **User Choice**: Users can choose between offline installation (models downloaded on first run) or full installation with models included.
- **Bandwidth Optimization**: Reduces initial download requirements for users with limited bandwidth, allowing them to get started faster.
- **Error Recovery**: Robust error handling for network issues, with clear user feedback and retry mechanisms.

---

## Version 3.0

**Release date:** 16 July 2025

### 1. Major Features & Core Capabilities

#### 1.1 **AI-Powered Face Restoration (GFPGAN)**

- **New `AI_face_restoration` Class**: A new, specialized class has been implemented to handle face restoration models. This class is architected to manage the unique preprocessing and post-processing requirements of models like GFPGAN, distinct from standard upscaling models.
- **GFPGAN Model Integration**: The GFPGAN v1.4 model has been added to the AI model repository and is now selectable from the UI. It is listed under a new `Face_restoration_models_list` category. The main orchestrator (`upscale_orchestrator`) now detects when a face restoration model is selected and routes the task to the appropriate `AI_face_restoration` instance.
- **Specialized Processing Pipeline**: The new class introduces a dedicated pipeline for face enhancement. This includes resizing the input image to the model's required dimensions (e.g., 512x512 for GFPGAN), handling color channel conversions, and post-processing the output to restore the image to its original dimensions.

### 2. UI/UX Modernisation

#### 2.1 **Complete Thematic Redesign**

- The application has undergone a significant visual overhaul with a new, professionally designed color scheme to improve aesthetics and user comfort during long sessions. The new theme provides better contrast and a more modern look.

| Element           | New Value (v3.0)       | Old Value (v2.2)          |
| :---------------- | :--------------------- | :------------------------ |
| Background        | `#1A1A1A` (Deep Black) | `#000000` (Pure Black)    |
| App Name Color    | `#FF4444` (Bright Red) | `#FF0000` (Pure Red)      |
| Widget Background | `#2D2D2D` (Dark Grey)  | `#5A5A5A` (Grey)          |
| Accent/Border     | `#FFD700` (Gold)       | Gold & Red                |
| Button Hover      | `#FF6666` (Light Red)  | `background_color`        |
| Info Button       | `#B22222` (Dark Red)   | `widget_background_color` |

#### 2.2 **Enhanced Splash Screen**

- **Dynamic Progress Bar**: The splash screen now features a `CTkProgressBar` to provide visual feedback on the application's loading status, enhancing the startup experience.
- **Smooth Fade-Out Animation**: A new `fade_out` method using a cosine function has been implemented for a smooth, animated exit transition instead of an abrupt disappearance.
- **Improved Information Display**: The splash screen now prominently displays the application version number.

#### 2.3 **Redesigned and Resizable Message Boxes**

- The `MessageBox` class was significantly improved to handle large blocks of text, such as detailed error messages. It now implements a `CTkScrollableFrame`, ensuring that content is always accessible without forcing the dialog to an unmanageable size.
- The dialogs now have defined `minsize` and `maxsize` properties for better window management.

#### 2.4 **Improved UI Readability**

- The main AI model dropdown menu is now logically grouped by model type (Upscaling, Denoising, Face Restoration, Interpolation), with a `MENU_LIST_SEPARATOR` between categories. This makes it easier for users to find and select the appropriate AI model for their task.

### 3. Performance and Code Optimisation

#### 4.0 **Memory Optimisation with Contiguous Arrays**

- Widespread use of `numpy.ascontiguousarray` has been implemented across the codebase. This is applied during critical image handling steps in `AI_upscale.preprocess_image`, `AI_interpolation.concatenate_images`, and the new `AI_face_restoration.preprocess_face_image` class. This ensures data is aligned in memory, which can significantly speed up operations in backend libraries like OpenCV and ONNX Runtime.

#### 3.2 **Refined Data Type Handling**

- The `AI_upscale` class now explicitly ensures input images are converted to `float32` before normalization, improving precision and preventing potential data type mismatches during inference.
- The `AI_face_restoration` class is configured to intelligently select between `float16` and `float32` based on the specific model's requirements (`fp16: True` in config), further optimizing performance and VRAM usage for compatible models.

### 4. Codebase Health and Maintainability

#### 4.1 **Specialised Class for Face Restoration**

- The logic for face restoration has been fully encapsulated within the new `AI_face_restoration` class, separating it from the general-purpose `AI_upscale` class. This object-oriented approach makes the code more modular, readable, and easier to extend with different face enhancement models in the future.

#### 4.2 **Robust BGRA to BGR Conversion**

- The application now explicitly handles images with an alpha channel (4-channel BGRA) when using face restoration models. A new import for `COLOR_BGRA2BGR` was added, and it is used within `preprocess_face_image` to convert images to the 3-channel BGR format expected by the GFPGAN model. This prevents runtime errors and ensures correct processing of PNGs or other images with transparency.

---

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

4.0 **Resolved Video Encoding Race Condition**

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

--

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

4.0 **Improved Code Organisation**

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

4.0 **Model List Structure**

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
