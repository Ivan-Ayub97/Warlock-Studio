_Warlock-Studio | Changelog History_

# Version 5.1

**Release date:** December 11, 2025
**Kernel Version:** 5.1.0 (Codename: Asynchronous-Warlock)
**Architecture:** Distributed / Multi-Threaded UI

### Asynchronous I/O Architecture & File Queue System

The application's file ingestion and management layer has been architecturally decoupled from the main GUI thread. The legacy synchronous file loader has been deprecated in favor of a reactive, non-blocking subsystem driven by the new **`file_queue_manager.py`** module.

#### Multi-Threaded Ingestion Pipeline (`FileQueueManager`)

- **Parallelized Metadata Extraction:** Implemented a `ThreadPoolExecutor` worker pool strategy. Heavy I/O operations—including `cv2.VideoCapture` metadata reading, frame extraction, and resolution calculation—are now offloaded to background threads, eliminating UI freezes ("Application Not Responding") during batch imports.
- **Thread-Safe Message Bus:** A `queue.Queue` messaging pattern was established to marshal data between worker threads and the main Tkinter loop. The `_check_queue` polling mechanism ensures atomic widget updates, preventing race conditions and segmentation faults common in multi-threaded GUI applications.
- **Object-Oriented State Encapsulation:** Introduced the `QueueItem` class to encapsulate file-specific state (loading status, error flags, thumbnail blobs, and resolution metrics). This allows for granular error handling where a single corrupt file no longer halts the entire batch loading process.
- **Lazy Thumbnail Generation:** Video and image thumbnails are now generated and cached asynchronously using a secure `PIL` <-> `OpenCV` bridge, reducing initial memory footprint during large batch operations.

#### Viewport State Management (Main Orchestrator)

- **Dynamic View Switching:** The main application logic now implements explicit view controllers (`show_drop_zone` and `show_file_manager`) to seamlessly toggle between the "Drag & Drop" landing state and the active "Queue Management" interface.
- **Drop Zone Expansion:** The drag-and-drop event listeners have been registered across the entire container hierarchy (`drop_zone_frame`, buttons, labels), eliminating "dead zones" where file drops were previously ignored.

### Preferences System Re-engineering

The configuration subsystem (`warlock_preferences.py`) has been rewritten from the ground up, abandoning the linear vertical scrolling design for a categorized **Sidebar Navigation Architecture**.

#### Diagnostic & Maintenance Suite

- **Integrated Log Viewer:** Embedded a read-only `CTkTextbox` that dynamically streams the content of the active `warlock_studio.log` file. Users can now audit FFmpeg stderr output and internal errors directly within the GUI without navigating the host file system.
- **Debug Package Export:** Introduced the `export_debug_info` routine. This function aggregates the current JSON configuration, recent log files, and system hardware telemetry into a timestamped `.zip` archive, streamlining the bug reporting process for developers.
- **Binary Path Overrides:** Added manual path selectors for `ffmpeg.exe` and `exiftool.exe`. This resolves path resolution conflicts on systems with non-standard environment variables or portable deployments.

#### Hardware & Execution Configuration

- **Strict ONNX Provider Enforcement:** Implemented a user-accessible selector for the AI Execution Backend (`Auto`, `CUDA`, `DirectML`, `CPU`). This allows users to forcefully bypass the internal driver heuristic engine if automatic detection fails on hybrid GPU setups.
- **Refined "NEO Engine":** The hardware scanning logic was optimized to prioritize `GPUtil` direct querying for NVIDIA GPUs while maintaining a robust WMI/WQL fallback chain for AMD and Intel architectures, ensuring accurate VRAM reporting.

### AI Algorithm Precision & Mathematical Corrections

#### RIFE Dynamic Padding (Artifact Elimination)

Resolved a critical mathematical deficiency in the RIFE (Real-Time Intermediate Flow Estimation) interpolation pipeline where input resolutions not divisible by 32 generated black banding or green-screen artifacts.

- **Reflective Padding Logic:** Implemented `pad_image_to_divisor` within the `AI_interpolation` class. This routine calculates the modulo-32 remainder and dynamically applies reflective border padding (`cv2.copyMakeBorder`) to the input tensor before inference.
- **Surgical Post-Inference Cropping:** A corresponding `crop_padding` routine removes the synthetic borders from the output tensor with pixel-perfect precision, ensuring the generated frames match the original resolution exactly without visual distortion.

#### Face Restoration Stabilization

- **Float32 Precision Enforcement:** The `AI_face_restoration` class now strictly enforces `float32` precision. The experimental `fp16` auto-detection was removed to prevent `NaN` (Not a Number) tensor propagation observed on specific CUDA driver versions and older Pascal/Maxwell architectures.
- **Alpha Channel Integrity Protocol:** Refined the RGBA handling pipeline ("Split-Merge Strategy"). The Alpha channel is now strictly processed via Bicubic interpolation, while only the RGB channels undergo GAN-based restoration. This prevents the generation of noise artifacts in transparency masks.

### Operational Safety & Visual Identity

#### Process Execution Guards

- **Confirmation Interceptor:** Injected a `messagebox.askyesno` guard clause into the `upscale_button_command`. This prevents accidental initiation of resource-intensive workloads.
- **Shutdown Race Condition Fix:** The `on_app_close` handler now captures window attributes (`is_topmost`) _before_ the destruction sequence begins, resolving `TclError: application has been destroyed` exceptions that previously corrupted preference saving on exit.

#### "Warlock Identity" Thematic Unification

- **High-Contrast Palette:** The interface theme has been standardized to a "Dark & Gold" palette. Backgrounds have been deepened to `#0A0A0A` (Near Black) to maximize contrast for image previewing, with semantic color coding (Success Green, Error Red, Warlock Gold) applied consistently across the new Preferences and Queue modules.

---

# Version 5.0

**Release date:** November 22, 2025
**Kernel Version:** 5.0.0 (Codename: NEO-Refactor)
**Architecture:** Modular / Component-Based

### Architectural Re-engineering & System Decoupling

The application core has undergone a foundational transformation, migrating from a legacy monolithic script architecture (v4.3) to a distributed, component-oriented modular system. This refactoring significantly reduces cyclomatic complexity, enforces Separation of Concerns (SoC), and isolates failure domains for improved fault tolerance.

#### Component Atomization & Isolation

The legacy `Warlock-Studio (1).py` codebase has been segmented into specialized operational subsystems:

- **`Warlock-Studio.py` (Core Orchestrator):**

  - Retains exclusive control over the application entry point, the `App` class instantiation, and the main GUI event loop (`mainloop`).
  - Manages the high-level orchestration of `multiprocessing.Process` spawning for AI workloads, ensuring UI thread non-blocking behavior.

- **`warlock_preferences.py` (State & Hardware Abstraction Layer):**

  - A newly isolated subsystem responsible for persistent state management via JSON serialization.
  - Hosts the **"NEO Engine"**, a heuristic hardware diagnostic suite.
  - Manages OTA (Over-The-Air) update logic via asynchronous polling of the GitHub Releases API.

- **`console.py` (I/O Stream Management):**

  - Encapsulates the `IntegratedConsole` widget logic and the `StreamRedirector` class.
  - Implements a thread-safe **Singleton Pattern** (`ConsoleManager`) to intercept standard output (`sys.stdout`) and error streams (`sys.stderr`) at the interpreter level, redirecting them to the GUI buffer in real-time.

- **`drag_drop.py` (OS Event Wrapper):**
  - Provides an abstraction layer over `tkinterdnd2`.
  - Implements the `DnDCTk` class, which inherits from `CTk` and `TkinterDnD.DnDWrapper`, injecting native OS file drag-and-drop event listeners into the custom UI toolkit.

### "NEO Engine": Telemetry, Heuristics & Persistence

#### Advanced Hardware Introspection (`HardwareScanner` Class)

Implemented a robust Hardware Abstraction Layer (HAL) within `warlock_preferences.py`, utilizing `wmi`, `psutil`, and `GPUtil` libraries for deep system profiling:

- **CPU Topology Analysis:** Differentiates between physical and logical cores to optimize `cpu_count` allocation for FFmpeg encoding and frame extraction threads.
- **Memory Mapping:** Performs real-time analysis of total vs. available RAM to preemptively throttle batch sizes and prevent page file thrashing (swapping).
- **GPU Heuristic Detection:** Implements a deterministic priority chain for VRAM detection:
  1. **Priority A:** `GPUtil` (Direct NVIDIA API access).
  2. **Priority B:** `WMI` (Win32_VideoController query for AMD/Intel).
  3. **Fallback:** Synthetic estimation based on system heuristics.
- **Algorithmic Optimization (`get_smart_recommendations`):** A logic engine that accepts hardware telemetry and computes the **"Safe VRAM Limit"** via the formula `max(0.5, vram_gb - 1.5)`. It dynamically prescribes Tile Resolutions and Thread Concurrency based on hardware tiers (Low-End vs. High-End).

#### Serialized Configuration Management (`ConfigManager`)

- **JSON State Persistence:** Deprecated hardcoded global variables in favor of a structured `warlock_config.json` file.
- **Robust Serialization:** The `ConfigManager` handles the marshaling and unmarshaling of user preferences (UI scaling factors, window opacity, process priority).
- **Integrity Validation:** The `load_config` method implements schema validation, automatically injecting default key-value pairs if the configuration file is corrupted or missing fields during boot.

### AI Inference Engine Stabilization (ONNX Runtime Backend)

#### Deterministic Session Initialization

- **Strict Type Enforcement:** The `create_onnx_session` factory was hardened to enforce strict integer typing (`int`) for the `device_id` parameter.
  - _Correction:_ In v4.3, passing device IDs as string literals (e.g., `"0"`) caused silent initialization failures on strict `DirectML` backends. The new logic creates an explicit mapping table (`'GPU 1' -> 0`).
- **Hierarchical Execution Priority:** Implemented a failover chain for `ExecutionProviders` to maximize hardware acceleration compatibility:
  1. **Primary:** `CUDAExecutionProvider` (NVIDIA Optimized via cuDNN).
  2. **Secondary:** `DmlExecutionProvider` (DirectX 12 Abstraction Layer).
  3. **Failover:** `CPUExecutionProvider` (Universal x64 instruction set).

#### Precision Standardization in Face Restoration (`AI_face_restoration`)

- **Mandatory Float32 Inference:** The experimental `fp16` (half-precision) auto-detection logic used in v4.3 was deprecated.
  - _Rationale:_ It caused `NaN` (Not a Number) tensor propagation on older GPUs lacking dedicated Tensor Cores. The engine now defaults to **`float32`** unless the ONNX model metadata explicitly mandates a lower precision dtype.
- **"Split-Merge" Channel Strategy:** A sophisticated pipeline was developed to handle RGBA (Transparency) artifacts:
  1. **Channel Splitting:** The Alpha channel is isolated from the RGB tensor.
  2. **Hybrid Scaling:** RGB channels are processed via the Neural Network; the Alpha channel is scaled using **Bicubic Interpolation** (`INTER_CUBIC`) to preserve edge fidelity without introducing GAN-generated noise into the transparency mask.
  3. **Tensor Reconstruction:** Post-inference concatenation (`numpy.concatenate`) reassembles the final image tensor.

#### Heuristic OOM (Out-Of-Memory) Recovery

- **Recursive Dynamic Tiling:** The `upscale_video_frames_async` pipeline now wraps inference calls within a specialized `try...except` block listening for specific `cuda`, `memory`, or `allocation` exceptions.
- **Recovery Algorithm:** Upon trapping a VRAM overflow event, the orchestrator intercepts the crash, dynamically downscales the `max_resolution` (Tile Size) by a factor of 2 (e.g., `100% -> 50%`), and triggers a recursive retry of the failed frame. This ensures batch job completion even during transient memory spikes.

### Video Processing Pipeline & FFmpeg Orchestration

#### Redundant Encoding Pipeline (Codec Fallback Loop)

The `video_encoding` function has been refactored into a transactional state machine with automatic rollback and retry capabilities:

- **Attempt 0 (Hardware Acceleration):** Initiates encoding using the user-selected hardware codec (e.g., `hevc_nvenc`, `h264_amf`, `hevc_qsv`).
- **Failure Handling:** Captures `subprocess.CalledProcessError` or non-zero exit codes (indicative of driver timeouts or resource locking).
- **Attempt 1 (Software Failover):** Automatically switches context to the CPU-based `libx264` encoder. This guarantees output file delivery even in catastrophic GPU driver failure scenarios.

#### Lossless Intermediate Serialization

- **PNG Standardization:** The `extract_video_frames` function has strictly deprecated the use of `.jpg` (v4.3) for temporary frame storage. The pipeline now mandates **`.png`** containers, eliminating compression artifacts (generation loss) before the pixel data enters the neural network input layer.

#### Temporal Frame Synchronization

- **FPS Multiplier Logic:** An explicit `fps_multiplier` argument was injected into the `video_encoding` signature.
  - _Functionality:_ This allows the orchestrator to mathematically calculate the target framerate (`source_fps * multiplier`) specifically for **RIFE Interpolation** tasks (x2, x4, x8), ensuring frame-perfect Audio/Video synchronization in the final container muxing.

### UI/UX & Advanced Diagnostic Tools

#### Integrated Debugging Console (`IntegratedConsole`)

- **Stream Interception:** The `StreamRedirector` class hooks into the Python interpreter's I/O layer to capture `sys.stdout` and `sys.stderr`.
- **Real-Time Visualization:** A dedicated `CTkTextbox` widget renders logs with Regex-based syntax highlighting (INFO=Blue, ERROR=Red, WARNING=Yellow, SUCCESS=Green), empowering end-users to diagnose underlying FFmpeg or ONNX runtime issues without external CLI tools.

#### 5.2. Dynamic DOM Adaptability (FluidFrames Integration)

- **Contextual Widget Injection:** Implemented the `clear_dynamic_menus` routine to manipulate the widget tree (DOM) at runtime.
  - _Behavior:_ Upon selecting a **RIFE** model, "Blending" controls are destroyed and "Frame Generation" selectors (Slowmotion factors) are dynamically injected into the layout. This prevents invalid configuration states at the UI level.

#### Window Management & Responsive Layout

- **Reactive Geometry Engine:** The `window.resizable(True, True)` flag was enabled, and the initial viewport geometry was increased to **85%** of screen height. The `grid` and `place` geometry managers were recalibrated to respond dynamically to window resizing events, accommodating the new bottom-docked console.

### Deployment & System Integration Strategy

#### Relocation of Default Installation Directory

- **Privilege Escalation Mitigation:** The default installation target within the Inno Setup script (`.iss`) has been migrated from `%ProgramFiles%` to **`%userprofile%\Documents\Warlock-Studio`**.
- **Write-Access Assurance:** This critical deployment change addresses `PermissionDenied` (WinError 5) exceptions encountered in v4.3 on systems with strict UAC (User Account Control) policies. By deploying to the user space, the application guarantees atomic read/write access for:
  - Serialization of the **`warlock_config.json`** file.
  - Real-time log appending to the **`Warlock_Logs`** directory.
  - Execution of the internal **PDF Manual** without invoking elevated Administrator privileges.
  - Unimpeded execution of the **NEO Engine** hardware probes without file system virtualization or sandbox interference.

---

# Version 4.3

**Release date:** November 15, 2025

### Critical Stability & Process Synchronization

#### Forced Writer Thread Synchronization (Critical Stability Fix)

- Implemented explicit and mandatory synchronization using the `t.join()` function for **all frame writing threads** (`writer_threads`) within the main video orchestrator (`upscale_orchestrator`).
- This change is **critical for workflow integrity**, ensuring that all AI-upscaled frames are completely written and closed on disk before initiating the FFmpeg video encoding process.
- **Technical Impact:** Resolves intermittent encoding failures (`File Not Found` or `Input/Output error`) that could occur when FFmpeg attempted to read files still being written by concurrent threads.

#### Graceful Termination for Process Monitoring Thread

- The status monitoring thread (`upscale_monitor_thread`) has been enhanced to handle unexpected failures or abrupt terminations of the main upscaling process.
- An explicit `break` statement was added to the `except Exception` block that manages errors reading from the status queue (`process_status_q`).
- **Technical Impact:** If the main upscaling process (`Process`) fails or hangs, the monitoring thread now **terminates its execution gracefully and immediately** instead of becoming a "zombie thread," allowing the user interface to correctly reset global status controls.

#### Reinforced Frame Sequence Integrity Validation

- A robust verification function was introduced to validate the integrity of the frame sequence on disk before starting the AI upscaling.
- This measure is vital for `Resume` operations and now explicitly validates:
  - The **physical existence** of every frame path (`os_path_exists`).
  - The **readability** of the image file (`image_read`), preventing segmentation faults or memory errors if the file is corrupted or incomplete.

### Performance Improvements and Technical Refinements

#### Aggressive Post-Process Memory Optimization

- A memory optimization step (`optimize_memory_usage`) with calls to `gc.collect()` was implemented after the last upscaled frames are saved to disk and before video encoding starts.
- **Purpose:** To reduce peak RAM usage by aggressively freeing any residual NumPy arrays and image references held in memory, ensuring the system has maximum memory available for the FFmpeg encoding subprocess.

#### Clarification of Tiling and VRAM Logic

- Detailed internal documentation (`# comments`) was added to the validation function (`check_upscale_settings`) to **explain the calculation logic for tile size** based on the user's VRAM limiter and the AI model's usage factor.
- **Clarified Formula:** `tiles_resolution = (ModelUsageFactor * VRAM_GB) * 100`. This is a technical refinement ensuring transparency and justification for the internal calculation.

#### Robustness and Clarification in Video Encoding (FFmpeg Command Line) 🎬

- **Explicit Stream Mapping:** Within `create_video_with_ffmpeg`, the FFmpeg command line now uses **more explicit and safer stream mapping** (`-map 0:v:0 -map 0:a:0?`). This ensures that the first video stream is selected obligatorily, and the first audio stream is selected optionally, resolving ambiguity issues in inputs containing multiple video or audio streams.
- **Data Stream Exclusion:** The **`-dn`** (Disable Data stream) argument was introduced into the base FFmpeg command line. This instructs the encoder to ignore unnecessary _data streams_ (such as subtitles or complex metadata) that are not relevant to the video or audio, **simplifying the muxing process and improving compatibility** across various players.

### Aesthetic and UI/UX Refinements

#### Complete Aesthetic Overhaul: Crimson Gold Dark Theme 🌑

- The application's entire color scheme was replaced (migrating from a v4.2.1 red/yellow palette) with a high-contrast **"Crimson Gold Dark Theme"**:
  - **Background:** Deep Black (`#121212`).
  - **Widgets/Panels:** Very dark Wine Red (`#4B0000`).
  - **Primary Accent:** Pure Gold (`#FFD700`).
- Dropdown menus (`CTkOptionMenu`) were styled with explicit **`corner_radius=0`** for a more modern, defined, and flat aesthetic.

#### Detailed Visualization of File Information in Queue

- Clarity of information in the queue widget (`FileWidget.add_file_information`) was improved by breaking down the complete resolution transformation sequence into three lines, eliminating ambiguity about intermediate dimensions:
  - **AI Input:** Shows the final input resolution.
  - **AI Output:** Shows the native resolution after upscaling by the AI model factor.
  - **Video Output:** Shows the final resolution of the encoded video.

#### Visual Consistency and Updated Menu Separators

- The visual separator in all dropdown menus has been updated from the string `"<------------------>"` (v4.2.1) to a cleaner, more discreet sequence of dots: **`• • • • • • • • • • • •`**.

#### Refined Dynamic Menu Logic

- The conditional visibility logic for menus in `select_AI_from_menu` was refined for a smoother user experience:
  - The **Frame Interpolation** control is only visible for RIFE models.
  - The **Blending** control is only visible for Upscaling/Facial Restoration models (with automatic disabling logic for GFPGAN).

---

# Version 4.2.1

**Release date:** 27 October 2025

### Critical Bug Fixes & Core Functionality

#### Resolved Critical Audio Passthrough Failure in Video Encoding

- Addressed a major bug introduced in v4.2 where all generated videos were encoded without audio. This was caused by the accidental omission of the `ffprobe.exe` executable (a key component of the FFmpeg suite) from the application's `Assets` directory and the final build package.
- The `video_encoding` pipeline relies on media analysis (performed by `ffmpeg` or `ffprobe`) to detect the presence of audio streams in the source video. Due to the missing executable, the `audio_info_command` subprocess would fail, causing the application to incorrectly assume all input videos had no audio.
- By restoring the `ffprobe.exe` binary to the application bundle, the audio detection step now executes successfully, enabling the proper copying (`-c:a copy`) or re-encoding (`-c:a aac`) of the original audio track into the final processed video.

#### Fixed Invalid UI State Persistence

- Addressed a critical UI logic bug where AI model settings could persist incorrectly when switching between model types. The `select_AI_from_menu` function now properly resets conflicting global variables:
  - **RIFE Selection:** When an interpolation model (e.g., RIFE, RIFE_Lite) is selected, the **`selected_blending_factor`** is now automatically reset to `0` (OFF), as blending is not a valid operation for this model.
  - **Upscaling/Face Selection:** Conversely, when any upscaling or face restoration model (e.g., BSRGAN, GFPGAN) is selected, the **`selected_frame_generation_option`** is automatically reset to `OFF`.
- This prevents users from applying invalid or non-functional setting combinations, ensuring a more intuitive and error-free workflow.

#### Fixed File Thumbnail Generation Crash

- Reworked the `FileWidget.extract_file_icon` method to be more robust during the conversion of an OpenCV (NumPy) image to a `CTkImage` for file previews.
- The process no longer relies on the `mode='RGB'` parameter within `pillow_image_fromarray`, which could fail with certain `numpy` array layouts.
- It now uses an explicit, two-step conversion (`pil_img = pillow_image_fromarray(source_icon)` followed by `pil_img = pil_img.convert("RGB")`), ensuring correct channel order and preventing potential `Pillow`/`numpy` compatibility crashes when loading file thumbnails.

### UI Corrections & Usability Enhancements

#### Corrected Supported File Extension List

- The helper text on the file selection screen has been corrected to more accurately list the supported file extensions. It now properly includes `jpeg` and `tiff` and removes incorrect entries (`heic`, `gif`, `mpg`, `qt`, `3gp`) to align with the application's actual processing capabilities.

#### Implemented Fixed Application Window

- The main application window is now non-resizable (`window.resizable(False, False)`). This change was implemented to ensure a stable and predictable UI layout, preventing the relative-coordinate-based widget placement (`relx`, `rely`) from breaking, overlapping, or scaling improperly when the window is resized.

#### Improved Dialog Window Behavior

- Removed the `self.attributes("-topmost", True)` flag from the `MessageBox` class (which handles error and info dialogs). This stops popup windows from aggressively forcing themselves to be the top-most window on the entire desktop. Dialogs now behave as standard application-modal windows, resolving a key usability issue where they could "steal focus" or obstruct other programs.

### Thematic & Visual Redesign

#### New "Inferno" Thematic Redesign

- Version 4.2.1 introduces a significant visual overhaul, moving away from the previous gold-and-grey theme to a high-contrast, dark-mode "inferno" theme. This new palette uses a deep red background with bright yellow and white text, designed to improve readability and reduce eye strain.

The color scheme has been updated as follows:

| Element               | New Value (v4.2.1)        | Old Value (v4.2)       |
| :-------------------- | :------------------------ | :--------------------- |
| **Background**        | `#480B0B` (Dark Red)      | `#1A1A1A` (Deep Black) |
| **App Name Color**    | `#FFFFFF` (White)         | `#DFDFDF` (Light Grey) |
| **Widget Background** | `#252525` (Near Black)    | `#2D2D2D` (Dark Grey)  |
| **Main Text**         | `#FFE32C` (Bright Yellow) | `#FFFFFF` (White)      |
| **Accent/Border**     | `#FFFFFF` (White)         | `#FFD700` (Gold)       |
| **Info Button**       | `#A80000` (Medium Red)    | `#B22222` (Dark Red)   |
| **Warning Color**     | `#E02CDA` (Magenta)       | `#FF8C00` (Orange)     |
| **Error Color**       | `#070087` (Dark Blue)     | `#DC143C` (Crimson)    |

#### Monospaced Typography Shift

- The application's global font has been changed from `"Segoe UI"` to `"Consola"`. This provides a more uniform, technical aesthetic across all UI elements, enhancing readability for file names, numeric values, and status messages.

#### Dropdown Menu Readability

- Updated the visual separators in dropdown menus from `----` to `<------------------>` for clearer, more pronounced grouping of model categories.

---

# Version 4.2

**Release date:** 6 October 2025

### Core Architecture & Performance Engineering

#### Advanced ONNX Runtime Integration & Execution Provider Strategy

- **Intelligent Provider Prioritization & Fallback Mechanism**: The core AI model loading architecture has been fundamentally re-engineered for superior performance and resilience. The new implementation leverages a prioritized provider list (`['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']`) within the `onnxruntime.InferenceSession` constructor. It first attempts to initialize a session using the **`CUDAExecutionProvider`**, which offers the highest performance by interfacing directly with NVIDIA's CUDA cores and Tensor Cores. If this fails (due to lack of an NVIDIA GPU, driver issues, or CUDA toolkit incompatibility), the system gracefully falls back and attempts initialization with the **`DmlExecutionProvider`**, utilizing Windows' DirectML API for broader hardware acceleration across various GPU vendors (NVIDIA, AMD, Intel). As a final failsafe, if no GPU acceleration provider can be initialized, it defaults to the **`CPUExecutionProvider`**, ensuring the application remains functional on any machine, albeit with reduced performance.
- **Centralized Session Management & Code Refactoring**: A new, unified function, `create_onnx_session`, has been introduced to abstract and centralize all ONNX session creation logic. This eliminates the redundant and error-prone boilerplate code that was previously duplicated within the `_load_inferenceSession` method of each individual AI class (`AI_upscale`, `AI_interpolation`, `AI_face_restoration`). This refactoring adheres to the **Don't Repeat Yourself (DRY)** principle, significantly improving code maintainability, reducing the surface area for bugs, and ensuring a consistent and robust model loading strategy across the entire application.
- **Enhanced Error Handling & Diagnostics**: The `try...except` block encapsulating the session creation loop is now more sophisticated. It logs specific warnings when a provider fails to initialize and clearly indicates which provider it is falling back to. This provides transparent diagnostics that are critical for troubleshooting performance issues related to hardware acceleration.

#### PyInstaller Packaging & Distribution Overhaul

- **Aggressive Dependency Pruning for Size Optimization**: The PyInstaller `.spec` file has been meticulously optimized to drastically reduce the final distribution size. An extensive `excludes` list has been implemented to explicitly instruct PyInstaller's analysis engine to ignore and not bundle large, non-essential packages. This prevents the recursive inclusion of entire scientific and machine learning ecosystems like `torch`, `transformers`, `matplotlib`, `pandas`, `scipy`, and `PyQt5`, which are often pulled in as sub-dependencies but are not required for this application's runtime. This strategic pruning reduces the package size by hundreds of megabytes.
- **Robust Hidden Import Declaration for Runtime Stability**: The `.spec` file now includes a `hiddenimports` list containing `onnxruntime.capi._pybind_state`, `onnxruntime.providers`, and `moviepy.editor`. This is critical for ensuring runtime stability, as these modules are often loaded dynamically (e.g., via `__import__` or C extensions) in a way that PyInstaller's static analysis cannot detect. Explicitly declaring them forces their inclusion, preventing `ModuleNotFoundError` crashes when the packaged application attempts to access these components.
- **Strategic Shift to "One-Folder" Distribution**: The build strategy has been transitioned from a "one-file" mode to a more robust and efficient "one-folder" mode. This is achieved by using the `COLLECT` block in the `.spec` file. Instead of bundling all dependencies into a single large executable that must be decompressed to a temporary directory (`_MEIPASS`) on every launch, the "one-folder" approach places the executable alongside its required `.dll`, `.pyd`, and data files. This results in significantly faster application startup times and avoids potential conflicts with antivirus software or system permissions related to executing from temporary locations.

### Installer & Distribution Strategy Refinement

#### Transition to a Full Offline Installer Model

- **Self-Contained & Hermetic Package**: The application's distribution model has been fundamentally shifted from a lightweight online/web installer to a comprehensive **offline installer**. All required runtime assets, most notably the large AI model files (`.onnx`), are now bundled directly within the Inno Setup executable. This creates a fully self-contained package that guarantees a successful installation without any external dependencies.
- **Elimination of Network Dependencies & Points of Failure**: This architectural change enhances installation reliability exponentially. It completely removes the dependency on an active internet connection during setup and eliminates critical points of failure, such as GitHub/SourceForge server downtime, network interruptions, firewalls, or changes in download URLs. The user is assured of a complete, atomic installation from a single authoritative file.
- **Radical Simplification of Installer Logic**: As a direct result of bundling all assets, the entire Pascal Script `[Code]` section in `Setup.iss` has been **completely removed**. This eliminates dozens of lines of complex logic responsible for creating custom download pages, handling HTTP requests, managing user cancellations, and, most critically, invoking external processes like PowerShell for archive extraction (`Expand-Archive`). This simplification makes the installer script significantly more robust, predictable, and easier to maintain.
- **Streamlined User Experience**: The installer's UI has been simplified by removing the "Download AI Models" task from the `[Tasks]` section. This avoids user confusion and streamlines the setup process, as the action is no longer necessary. The output filename is also now explicitly suffixed with `-Full-Installer` to clearly communicate its offline nature.

### UI, UX & Developer Experience Enhancements

#### Enhanced Application Startup & Initial Feedback

- **Professional Splash Screen Implementation**: A new `SplashScreen` class provides immediate visual feedback upon application launch. This improves the _perceived performance_ by displaying a branded loading screen with status messages while core components and AI models are being initialized in the background. It prevents the appearance of a hung or unresponsive application during the initial loading phase.
- **Runtime Environment Diagnostics**: At startup, the application now programmatically queries and prints the available ONNX Runtime providers by calling `onnxruntime.get_available_providers()`. This information is outputted to the console, serving as an invaluable, zero-effort diagnostic tool. It allows both end-users and developers to instantly verify which hardware acceleration backends are detected and available to the application, aiding in performance tuning and troubleshooting.

#### Improved Debugging and Diagnosability

- **Persistent Console for Runtime Output**: The PyInstaller build configuration was modified to set `console=True`. This forces the application to run with an attached console window that captures the `stdout` and `stderr` streams. This is a critical enhancement for diagnostics, as it makes all print statements, logging output, warnings, and unhandled exception tracebacks immediately visible, providing a clear and persistent record of the application's runtime behavior for effective bug reporting and debugging.

---

# Version 4.1

**Release date:** 1 August 2025

### Model Enhancement and Utilization

#### Enhanced GPU Utilization and Error Handling

- **Robust Provider Configuration**: Added a private method `_select_providers` to intelligently select ONNX runtime providers based on the chosen GPU and improve model execution efficiency.
- **Dynamic Fall-back Mechanism**: Enhanced `_load_inferenceSession` to prioritize loading models on GPU providers and gracefully fallback to CPU providers if GPU initialization fails, coupled with detailed logging.
- **Improved Model Initialization**: Ensured comprehensive validation and error handling during model loading to enhance stability across various hardware environments.

### Compatibility and Runtime Improvements

#### Addressed NumPy and OpenCV Compatibility

- **Resolved Import Errors**: Fixed critical compatibility issues between NumPy 2.x and OpenCV by downgrading to NumPy 1.26.4, preventing `_ARRAY_API not found` and `numpy.core.multiarray failed to import` errors.
- **Critical Module Validation**: Ensured that all critical libraries (OpenCV, ONNX Runtime, CustomTkinter) load successfully without compatibility warnings, enhancing overall application reliability.
- **Runtime Environment Stability**: Resolved module loading conflicts that previously caused application crashes during startup.

### Performance Optimization

#### Memory and Resource Management

- **Contiguous Memory Utilization**: Enhanced use of `numpy.ascontiguousarray` throughout image processing pipelines to optimize memory usage during intensive AI processing tasks.
- **GPU Memory Error Recovery**: Improved handling of GPU memory allocation failures with automatic fallback to CPU processing when DirectML providers are unavailable.
- **Processing Pipeline Optimization**: Streamlined AI model loading and inference execution for better resource utilization across different hardware configurations.

### Code Quality and Error Handling

#### Enhanced Error Resilience

- **Syntax Error Resolution**: Fixed critical syntax error in `_load_inferenceSession` method that prevented proper AI model initialization.
- **Improved Error Messaging**: Enhanced logging capabilities with more informative error messages and warnings for better user diagnostics and troubleshooting.
- **Graceful Degradation**: Implemented improved fallback strategies for GPU resource issues, effectively preventing application crashes by dynamically adjusting processing pathways.
- **Provider Validation**: Added comprehensive validation for ONNX runtime providers with automatic fallback from GPU to CPU execution when hardware acceleration is unavailable.

### UI and User Experience

#### Application Stability and Feedback

- **Startup Reliability**: Resolved critical startup issues that prevented the application from launching due to module compatibility problems.
- **Processing Status Updates**: Enhanced real-time feedback during AI model loading and image/video processing operations.
- **Error Notification**: Improved error dialogs and status messages to provide clearer information about processing states and potential issues.
- **Hardware Compatibility**: Better detection and handling of different GPU configurations, with informative warnings when falling back to CPU processing.

### Technical Improvements

#### Model Loading Architecture

- **Modular Provider Selection**: Separated provider selection logic into dedicated methods for better code organization and maintainability.
- **Robust Model Validation**: Enhanced file existence checking and model integrity validation before attempting to load AI models.
- **Cross-Platform Compatibility**: Improved compatibility across different Windows configurations and GPU setups.

---

# Version 4.0.1

**Release date:** 27 July 2025

### Model Cleanup and Optimization

#### SuperResolution-10 Model Removal

- **Model Deprecation**: Removed the SuperResolution-10 model from the application due to performance and compatibility issues.
- **Code Cleanup**: Eliminated the dedicated `AI_super_resolution` class and all related processing pipelines.
- **UI Updates**: Removed SuperResolution-10 from model selection dropdown and information dialogs.
- **Memory Optimization**: Cleaned up VRAM usage configurations by removing SuperResolution-10 entries (0.8 GB allocation).
- **Streamlined Processing**: Simplified the upscaling orchestrator by removing SuperResolution-specific routing logic.
- **Model List Cleanup**: Removed `SuperResolution_models_list` from the main AI models collection.

#### Performance Improvements

- **Reduced Memory Footprint**: Application now uses less memory without the SuperResolution-10 model overhead.
- **Simplified Code Paths**: Cleaner processing logic with fewer conditional branches for model selection.
- **Enhanced Stability**: Removed potential failure points associated with the deprecated model.

---

# Version 4.0

**Release date:** 18 July 2025

### AI Model Integration & Enhancement

#### SuperResolution-10 Model Implementation

- **New AI Model Integration**: Added support for the SuperResolution-10 model, providing advanced super-resolution capabilities with 10x upscaling factor. This model is specifically designed for very low-resolution images and excels at significant resolution increases.
- **Specialized Processing Pipeline**: Created a dedicated `AI_super_resolution` class that inherits from the new `AI_model_base` class, implementing proper preprocessing (CHW format conversion, normalization) and postprocessing (HWC format conversion, value clipping) for optimal results.
- **Model Information Integration**: Added comprehensive model information in the AI model selector dialog, including year (2023), function (high-resolution image enhancement), and specialized use cases.

#### AI Architecture Improvements

- **Base Class Implementation**: Created the missing `AI_model_base` class that provides common functionality for all AI models, including ONNX model loading with GPU acceleration support and proper error handling.
- **Enhanced Model Loading**: Implemented robust model loading with provider selection (DML, CPU) and comprehensive error handling for missing model files.
- **VRAM Management**: Added VRAM usage information for SuperResolution-10 (0.8 GB) to help users optimize their GPU memory usage.

### Code Quality & Stability

#### Import System Optimization

- **Fixed Import Errors**: Resolved critical `NameError: name 'numpy_ndarray' is not defined` by properly organizing imports at the top of the file.
- **Consolidated Imports**: Removed duplicate import sections and properly structured the import hierarchy for better maintainability.
- **Type Annotation Fixes**: Corrected type annotations throughout the codebase to use the proper imported numpy types.

#### Enhanced Error Handling

- **Model Loading Resilience**: Implemented try-catch blocks for model loading operations with meaningful error messages.
- **Graceful Degradation**: Added fallback mechanisms that return the original image if super-resolution enhancement fails, ensuring the application never crashes.
- **Debug Information**: Enhanced logging with model loading status and error reporting for better troubleshooting.

### User Interface Updates

#### Model Selection Enhancement

- **Updated Model List**: SuperResolution-10 is now properly integrated into the AI model dropdown menu and categorized appropriately.
- **Information Dialog Updates**: Added detailed information about the SuperResolution-10 model in the help dialog, including its capabilities and recommended use cases.
- **Model Orchestration**: Enhanced the upscaling orchestrator to properly detect and route SuperResolution model tasks to the appropriate processing pipeline.

### Technical Improvements

#### Processing Pipeline Optimization

- **Specialized Image Processing**: Implemented dedicated image processing functions for super-resolution models that handle the unique requirements of the SuperResolution-10 model.
- **Memory Efficiency**: Optimized image preprocessing and postprocessing to minimize memory usage during super-resolution operations.
- **Performance Monitoring**: Added processing time tracking for super-resolution operations to help users understand processing performance.

#### Integration Completeness

- **Full Model Integration**: SuperResolution-10 is now fully integrated into all aspects of the application, from model selection to processing to output generation.
- **Consistent User Experience**: The super-resolution workflow follows the same patterns as other AI models, ensuring a consistent user experience.
- **Quality Assurance**: Implemented comprehensive testing to ensure the SuperResolution-10 model works correctly with both individual images and batch processing.

### Smart AI Model Distribution System

#### Automatic Model Download

- **Lightweight Installer**: Significantly reduced installer size from 1.4GB to approximately 300MB by removing AI models from the installation package.
- **On-Demand Download**: Implemented intelligent model downloading system that automatically fetches required AI models (327MB) when the application is first launched.
- **Progress Tracking**: Added visual progress indicators with download speed and completion percentage during model acquisition.
- **Fallback URLs**: Integrated multiple download sources (GitHub Releases, SourceForge) to ensure reliable model availability.
- **Resume Capability**: Download system supports resuming interrupted downloads and validates file integrity.

#### PyInstaller Optimization

- **Optimized Packaging**: Updated `.spec` file to exclude AI model directory from executable packaging, reducing final executable size by over 1GB.
- **Enhanced Dependencies**: Added model downloader module to the build process with proper hidden imports for requests, threading, and file handling libraries.
- **Improved Compression**: Increased optimization level and added module exclusions to further reduce executable size.

#### Installation Experience

- **Smart Setup Script**: Created enhanced Inno Setup configuration that can optionally download models during installation or defer to first-run.
- **User Choice**: Users can choose between offline installation (models downloaded on first run) or full installation with models included.
- **Bandwidth Optimization**: Reduces initial download requirements for users with limited bandwidth, allowing them to get started faster.
- **Error Recovery**: Robust error handling for network issues, with clear user feedback and retry mechanisms.

---

# Version 3.0

**Release date:** 16 July 2025

### Major Features & Core Capabilities

#### AI-Powered Face Restoration (GFPGAN)

- **New `AI_face_restoration` Class**: A new, specialized class has been implemented to handle face restoration models. This class is architected to manage the unique preprocessing and post-processing requirements of models like GFPGAN, distinct from standard upscaling models.
- **GFPGAN Model Integration**: The GFPGAN v1.4 model has been added to the AI model repository and is now selectable from the UI. It is listed under a new `Face_restoration_models_list` category. The main orchestrator (`upscale_orchestrator`) now detects when a face restoration model is selected and routes the task to the appropriate `AI_face_restoration` instance.
- **Specialized Processing Pipeline**: The new class introduces a dedicated pipeline for face enhancement. This includes resizing the input image to the model's required dimensions (e.g., 512x512 for GFPGAN), handling color channel conversions, and post-processing the output to restore the image to its original dimensions.

### UI/UX Modernisation

#### Complete Thematic Redesign

- The application has undergone a significant visual overhaul with a new, professionally designed color scheme to improve aesthetics and user comfort during long sessions. The new theme provides better contrast and a more modern look.

| Element           | New Value (v3.0)       | Old Value (v2.2)          |
| :---------------- | :--------------------- | :------------------------ |
| Background        | `#1A1A1A` (Deep Black) | `#000000` (Pure Black)    |
| App Name Color    | `#FF4444` (Bright Red) | `#FF0000` (Pure Red)      |
| Widget Background | `#2D2D2D` (Dark Grey)  | `#5A5A5A` (Grey)          |
| Accent/Border     | `#FFD700` (Gold)       | Gold & Red                |
| Button Hover      | `#FF6666` (Light Red)  | `background_color`        |
| Info Button       | `#B22222` (Dark Red)   | `widget_background_color` |

#### Enhanced Splash Screen

- **Dynamic Progress Bar**: The splash screen now features a `CTkProgressBar` to provide visual feedback on the application's loading status, enhancing the startup experience.
- **Smooth Fade-Out Animation**: A new `fade_out` method using a cosine function has been implemented for a smooth, animated exit transition instead of an abrupt disappearance.
- **Improved Information Display**: The splash screen now prominently displays the application version number.

#### Redesigned and Resizable Message Boxes

- The `MessageBox` class was significantly improved to handle large blocks of text, such as detailed error messages. It now implements a `CTkScrollableFrame`, ensuring that content is always accessible without forcing the dialog to an unmanageable size.
- The dialogs now have defined `minsize` and `maxsize` properties for better window management.

#### Improved UI Readability

- The main AI model dropdown menu is now logically grouped by model type (Upscaling, Denoising, Face Restoration, Interpolation), with a `MENU_LIST_SEPARATOR` between categories. This makes it easier for users to find and select the appropriate AI model for their task.

### Performance and Code Optimisation

#### Memory Optimisation with Contiguous Arrays

- Widespread use of `numpy.ascontiguousarray` has been implemented across the codebase. This is applied during critical image handling steps in `AI_upscale.preprocess_image`, `AI_interpolation.concatenate_images`, and the new `AI_face_restoration.preprocess_face_image` class. This ensures data is aligned in memory, which can significantly speed up operations in backend libraries like OpenCV and ONNX Runtime.

#### Refined Data Type Handling

- The `AI_upscale` class now explicitly ensures input images are converted to `float32` before normalization, improving precision and preventing potential data type mismatches during inference.
- The `AI_face_restoration` class is configured to intelligently select between `float16` and `float32` based on the specific model's requirements (`fp16: True` in config), further optimizing performance and VRAM usage for compatible models.

### Codebase Health and Maintainability

#### Specialised Class for Face Restoration

- The logic for face restoration has been fully encapsulated within the new `AI_face_restoration` class, separating it from the general-purpose `AI_upscale` class. This object-oriented approach makes the code more modular, readable, and easier to extend with different face enhancement models in the future.

#### Robust BGRA to BGR Conversion

- The application now explicitly handles images with an alpha channel (4-channel BGRA) when using face restoration models. A new import for `COLOR_BGRA2BGR` was added, and it is used within `preprocess_face_image` to convert images to the 3-channel BGR format expected by the GFPGAN model. This prevents runtime errors and ensures correct processing of PNGs or other images with transparency.

---

# Version 2.2

**Release date:** 7 July 2025

### Major Enhancements and Stability Overhaul

#### Comprehensive Logging System

- Implemented a full-featured `logging` system that writes to files in the user's `Documents` folder (`warlock_studio.log` and `error_log.txt`). This provides detailed diagnostics for debugging without relying solely on console output.
- A unified `log_and_report_error` function centralizes error handling, ensuring all critical issues are both logged and displayed to the user.

#### Proactive Environment Validation

- The application now performs pre-flight checks before processing begins to prevent common failures.
- Includes validation for Python version, required modules (`validate_environment`), FFmpeg availability, disk space, and available RAM (`validate_system_requirements`).
- Verifies that all input file paths exist and are accessible (`validate_file_paths`) and that the output directory is writable (`validate_output_path`).

#### Resilient Video Encoding Pipeline

- The entire `video_encoding` function was overhauled for maximum reliability.
- **Codec Fallback:** The system now tests for hardware codec availability (NVENC, AMF, QSV) before encoding. If a selected hardware encoder is not functional, it automatically falls back to the highly compatible `libx264` software encoder.
- **Robust Audio Handling:** Implements a fallback chain for audio processing. It first attempts to directly copy the audio stream; if that fails, it attempts to re-encode it; if that also fails, it finalizes the video without audio, ensuring a video file is always produced.

#### Graceful Shutdown and Cleanup

- Implemented `atexit` and `signal` handlers to ensure that temporary files are cleaned up and child processes are terminated safely, even on unexpected exits.
- Replaced abrupt `process.kill()` calls with the more graceful `process.terminate()` to allow for cleaner process shutdown.

### Performance and Memory Optimization

#### Aggressive Memory Management

- Video frame processing (`upscale_video_frames_async`) no longer holds large numbers of frames in RAM. It now writes small batches to disk and immediately calls the garbage collector (`gc.collect()`) to free memory, dramatically reducing the risk of crashes on long videos.

#### Dynamic GPU VRAM Error Recovery

- The AI orchestration logic can now recover from GPU "out of memory" errors during tiling. If an error is detected, it automatically reduces the tile resolution and retries the operation on that specific frame, preventing a total process failure.

### Critical Bug Fixes

#### Resolved Video Encoding Race Condition

- Fixed a critical bug where video encoding could start before all frame-writing threads were complete. The system now tracks all writer threads and explicitly waits for them to finish (`thread.join()`) before beginning the final video encoding, preventing corrupted or incomplete videos.

#### Corrected Persistent Stop Flag

- The `stop_thread_flag` is now reset (`.clear()`) at the start of each "Make Magic" execution, fixing a bug where a previously stopped job would prevent a new one from running.

#### Eliminated Status Update Race Condition

- Implemented a `threading.Lock` (`global_status_lock`) to protect shared flags that update the GUI. This prevents race conditions where multiple threads could attempt to modify the status simultaneously.

### UI / UX Refinements

#### Updated Splash Screen

- Reduced splash screen duration to 10 seconds for a faster application start-up.
- Corrected asset path to `Assets/banner.png` for proper display.

#### New Application Theme

| Element           | New Value        | Old Value (v2.1)     |
| :---------------- | :--------------- | :------------------- |
| App name          | `#FF0000` (Red)  | `#ECD125` (Gold)     |
| Widget background | `#5A5A5A` (Grey) | `#960707` (Dark Red) |
| Accent/Border     | Gold & Red       | Blue & Red           |

### Codebase Health and Maintainability

#### Enhanced Checkpointing and Recovery

- Added functions (`create_checkpoint`, `load_checkpoint`) to save and resume the progress of video frame processing, allowing recovery from interruptions.

#### Hardened Core Methods

- Core methods in AI classes now include checks for `None` inputs and feature default fallbacks (`case _:`) in `match` statements to prevent unexpected errors with unsupported data.

---

# Version 2.1

**Release date:** 23 June 2025

### Major Enhancements and Stability Overhaul

#### Robust Error Handling

- Model loading (`AI_upscale`, `AI_interpolation`) wrapped in `try…except FileNotFoundError, OSError`; meaningful error messages propagate to GUI.
- `extract_video_frames()` validates file existence, `cv2.VideoCapture.isOpened()`, and frame count > 0.
- `video_encoding()` captures `subprocess.CalledProcessError`, logs `stderr`, and continues with fallback strategies.
- Audio passthrough failures now trigger a silent audio‑less encode instead of total job abort.

#### Safe Thread and Process Management

- Deprecated error‑raising thread stop replaced with `threading.Event` (`stop_thread_flag`) polled at defined checkpoints.

#### Resilient Core Processing

- `copy_file_metadata()` now verifies `exiftool.exe` availability and the existence of source/target before execution.

### UI / UX Refinements

#### Refined Colour Palette

| Element           | New Value                       |
| ----------------- | ------------------------------- |
| App name          | `#ECD125`                       |
| Widget background | `#960707`                       |
| Active border     | Red (same as widget background) |

### Code‑base Maintainability

#### Improved Code Organisation

- File‑extension lists extracted to `filetypes.py` as `SUPPORTED_IMAGE_EXTENSIONS`, `SUPPORTED_VIDEO_EXTENSIONS`.

#### Dependency and Initialisation

- Added imports: `shutil.move`, `subprocess.CalledProcessError`, `threading.Event`.
- Global variables initialised in `init_globals()` for deterministic start‑up.

---

# Version 2.0

**Release date:** 6 June 2025

### Major Features

#### AI Frame Interpolation Support (`AI_interpolation` class)

- Supports RIFE‑based ONNX models; generates 1 (×2), 3 (×4), or 7 (×8) intermediate frames.
- Provides both real‑time preview and batch processing modes.
- Integrates with `FrameScheduler` for temporal upscaling pipelines.

#### RIFE Models Integration

- Added **RIFE** and **RIFE_Lite** to model repository.
- `RIFE_models_list` enumerates available checkpoints; `AI_models_list` now merges SRVGGNetCompact, BSRGAN, IRCNN, and RIFE families.

### Enhancements

#### Visual/UI Redesign

- Application renamed to **“Warlock‑Studio”** (with hyphen).
- New dark palette (`#121212`, `#454242`) with bright white text (`#FFFFFF`) and accent red (`#FF0E0E`).

#### Version‑Specific User Preferences

- User configuration stored as `Warlock-Studio_<major>.<minor>_UserPreference.json` to avoid backward‑compatibility clashes.

#### Modular and Scalable Layout System

- Added GUI constants defined in `layout_constants.py` (e.g., `OFFSET_Y_OPTIONS`, `COLUMN_1_5`).

#### Extended File‑Type Compatibility

- Updated `SUPPORTED_FILE_EXTENSIONS` and `SUPPORTED_VIDEO_EXTENSIONS` to include modern codecs (e.g., HEIC, AVIF, WebM).

#### Improved GPU Execution Support

- `provider_options` enumerates up to four DirectML devices (Auto, GPU 1 – GPU 4); selection persists across sessions.

### Technical Refinements

#### Model List Structure

- Menu drop‑downs now grouped by category separated by `MENU_LIST_SEPARATOR` for readability.

#### Advanced Interpolation Logic

- Implements tree‑based frame generation (e.g., D→A‑B‑C) with dependency tracking to avoid redundant inference passes.

#### Improved Numeric Precision and Post‑Processing

- Normalisation uses 32‑bit floats with epsilon guarding; RGBA conversion paths optimised using `numexpr`.

### UI / UX Refinements

#### Resizable Message Dialogs

- `MessageBox`; Tk `resizable(True, True)`.

#### Improved Dialog Formatting

- Uniform spacing, font hierarchy, and default‑value display.

### Minor Fixes

#### Code Corrections

- Corrected typo `ttext_color` → `text_color`.
- Expanded inline comments and reorganised sections for clarity.

---

# Version 1.1

**Release date:** 20 May 2025

### Major Improvements

#### Program Start‑up Optimisation

- Launch time reduced via lazy module loading.

#### Model Loading Improvements

- Parallel prefetch and checksum verification.

#### General Performance Optimisation

- Core refactor, improved I/O scheduling, and smarter resource allocation.

### Minor Fixes

#### Accessibility

- User‑interface tweaks for better accessibility (focus indicators, tab order).
