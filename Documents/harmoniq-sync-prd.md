# SyncStudio v1 — Product Requirements Document (PRD)

**Working name:** SyncStudio  
**Product type:** macOS desktop app (pro video/audio utility)  
**Primary goal:** Automatically synchronize multiple audio/video clips (PluralEyes-style) with high reliability, transparent confidence, and export to major NLEs.  
**Business model:** £99/year subscription (indie/edu discounts; team licensing later).  
**Platforms:** macOS 14+ (Sonoma) and macOS 15+ preferred; **Apple silicon first**; Intel optional if performance acceptable.  
**Technology direction:** SwiftUI + SwiftData app, C++17 core engine with a stable C ABI, AVFoundation-first decoding, optional FFmpeg in Direct (non‑MAS) build.  

---

## 1. Vision & Objectives

Create the most **reliable**, **fast**, and **transparent** A/V sync tool for macOS. Editors should trust the result, understand *why* it worked (or didn’t), and fix edge cases quickly. v1 lays a robust foundation for future add‑ons: on‑device transcription, cloud sync, watch folders, and automation.

### Success criteria (v1)
- **Accuracy:** P95 absolute sync error < **20 ms** across the golden dataset; median < **5 ms**.  
- **Drift:** P95 drift error < **2 ppm** on >30‑minute takes.  
- **Reliability:** Crash‑free sessions with resumable jobs; deterministic results across runs.  
- **Throughput:** ≥ **3× realtime** analysis on base M1 for typical speech/music content (single pair); ≥ **5×** on M2/M3.  
- **UX trust:** Clear confidence indicator and explainability (heatmap/overlap visualization).  

---

## 2. Users & Use Cases

### Personas
- **Indie Wedding Videographer (primary)**: Multi‑cam ceremonies; dual‑system audio; long takes; mixed sample rates.  
- **YouTube Creator/Podcaster**: Two cameras + external recorder; frequent batch jobs.  
- **Documentary Editor**: Long interviews (1–2h) with occasional clock drift; needs accurate time‑warp map.  
- **Post‑house Assistant Editor**: Batch pre‑sync for suites; wants CLI for repeatable pipelines.

### Core workflows
1. **Quick Sync**: Drop two or more clips → run → preview offsets & confidence → export to NLE.  
2. **Batch Sync**: Queue many pairs/projects, let it run, return to results later.  
3. **Manual Assist**: Inspect heatmap/waveforms, nudge an offset, lock a match, re‑run locally.  
4. **Export**: FCPXML / Premiere XML / Resolve CSV (or EDL) with offsets and optional time‑warp keyframes.

---

## 3. Scope

### In‑scope (v1)
- **macOS app** (SwiftUI + SwiftData) with project/job management.  
- **C++17 sync engine** behind a C ABI: coarse offset, fine alignment, drift estimation, confidence score, piecewise time‑warp keyframes.  
- **Decoding**: AVFoundation‑based audio extraction; resample and downmix to mono float32 (16–22.05 kHz).  
- **Exports**: FCPXML, Premiere XML, Resolve CSV (documented schema), plus CSV of keyframes.  
- **Job queue** with progress, cancel/resume, auto‑save/recover.  
- **Explainability UI**: confidence meter; per‑clip offsets; visual heatmap of overlap quality; manual nudge.  
- **CLI tool** that wraps the same core (for CI and pro pipelines).  
- **Two distributions**: Mac App Store (AVF‑only, sandboxed) and Direct (optional FFmpeg; Sparkle updates).  

### Future (post‑v1, considered but not required)
- **Transcription**: On‑device (Apple Speech or whisper.cpp) with timestamped captions.  
- **Watch folders** and automation rules.  
- **Team licensing**, cloud backup of project metadata, collaborative review.  
- **Plugin bridges** for direct NLE integration.  

### Out of scope (v1)
- Cloud‑hosted processing, web UI, Windows/Linux builds, GPU video decode, deep audio restoration.

---

## 4. Functional Requirements

### 4.1 Import & Project Management (SwiftUI + SwiftData)
- **R‑4.1.1**: Import files via drag‑and‑drop or file picker (NSOpenPanel).  
- **R‑4.1.2**: If sandboxed, capture **security‑scoped bookmarks** for persistent access.  
- **R‑4.1.3**: On import, probe duration, sample rate, channels; compute content hash/checksum (used to detect changes).  
- **R‑4.1.4**: Store Projects, Clips, Jobs, and Results in SwiftData models.  
- **R‑4.1.5**: Projects auto‑save; crash recovery restores to last stable state.

### 4.2 Sync Engine (C++17 core via C ABI)
- **R‑4.2.1 Decode**: Extract PCM (mono float32) at common sample rate (configurable, default 22050 Hz). Prefer AVFoundation; allow FFmpeg on Direct build.  
- **R‑4.2.2 Features**: Frame ~23 ms, hop 10 ms; pre‑emphasis & HPF; compute **spectral flux** and **log‑mel/chroma** features; compress for sidecar caching (.syncidx).  
- **R‑4.2.3 Coarse Alignment**: Multi‑resolution FFT cross‑correlation (GCC‑PHAT on flux) to find top‑K candidate offsets (±N minutes).  
- **R‑4.2.4 Fine Alignment**: Constrained **DTW** (Sakoe‑Chiba band) around candidates; pick best by combined cost.  
- **R‑4.2.5 Drift Estimation**: Windowed alignment every 30–60 s → robust line fit (RANSAC) → **drift ppm** and **piecewise linear time‑warp** keyframes.  
- **R‑4.2.6 Confidence**: Scalar 0..1 from peak sharpness, DTW cost, and envelope re‑correlation.  
- **R‑4.2.7 Output**: `baseOffsetMs`, `driftPpm`, `confidence`, `keyframes[]`.  
- **R‑4.2.8 Determinism**: Same inputs → identical result; fixed seeds; stable sort of parallel tasks.  

### 4.3 UI/UX
- **R‑4.3.1**: Project list; job detail with progress and logs.  
- **R‑4.3.2**: Visual **alignment heatmap**; show offsets & drift in clear language.  
- **R‑4.3.3**: Manual **nudge** (+/‑ ms) and ability to **lock** matches then re‑run locally.  
- **R‑4.3.4**: Non‑blocking operations; responsive during long jobs.  
- **R‑4.3.5**: Clear error states with suggested retries (speech‑focus, music‑focus, broader search).

### 4.4 Export
- **R‑4.4.1**: Export **FCPXML**, **Premiere XML**, **Resolve CSV/EDL** representing offsets and optional warp.  
- **R‑4.4.2**: Export **CSV**/JSON of alignment results (for audit and scripts).  
- **R‑4.4.3**: Validate exports against small known-good NLE projects.

### 4.5 CLI
- **R‑4.5.1**: `sync_cli A.wav B.wav` → JSON result on stdout.  
- **R‑4.5.2**: Options to write CSV/XML exports and configure params (SR, search window, mode).  
- **R‑4.5.3**: Used by CI to run golden dataset regression checks.

### 4.6 Distribution & Licensing
- **R‑4.6.1**: **MAS build** (sandboxed, notarized, AVFoundation‑only).  
- **R‑4.6.2**: **Direct build** (Sparkle updates; optional FFmpeg; different entitlements).  
- **R‑4.6.3**: License system for Direct build with £99/year subscription, offline grace period, indie/edu discounts.  

---

## 5. Non‑Functional Requirements

- **Performance**: Targets listed in §1.  
- **Reliability**: Crash‑safe temp handling; journaling of jobs; auto‑recovery; graceful cancellation.  
- **Determinism**: Identical outputs for identical inputs & params; stable across machines of same arch and OS.  
- **Scalability**: Handle long takes (≥ 2 hours) and many jobs in queue.  
- **Security/Privacy**: No external uploads in v1; local processing only. Hardened runtime; signed & notarized.  
- **Accessibility**: Basic VoiceOver labels; keyboard navigation.  
- **Internationalization**: English UI; foundations for localization later.  

---

## 6. Architecture Overview

### Components
- **Sync Core (C++17)**: Feature extraction, alignment, drift estimator, verification, sidecar I/O.  
- **C ABI**: `sync_core.h` exposing `sync_create/destroy`, `sync_feed_{a,b}`, `sync_run`, `sync_align_files`, `sync_last_error`.  
- **Swift Package (SyncKit)**: Thin wrapper providing async Swift APIs; marshals paths, progress, and results.  
- **macOS App**: SwiftUI views, SwiftData models, JobRunner orchestrating work via Swift concurrency.  
- **Decoders**: AVFoundation (Obj‑C++ bridge in `decoding_avf.mm`); optional FFmpeg (`decoding_ffmpeg.cpp`) for Direct build.  
- **CLI**: Command-line tool linked to the same core.

### Data
- **Project** (`.syncproj` concept held in SwiftData): references clips (URLs + bookmarks), parameters, results, export records.  
- **Sidecar** (`.syncidx`): compressed features (flux/chroma/log‑mel) for fast re‑runs.  
- **Result JSON/CSV**: offsets, drift, confidence, keyframes.  

### Build & Packaging
- CMake for core and CLI; SPM for Swift wrapper; Xcode for app. Separate MAS vs Direct schemes/entitlements.

---

## 7. Detailed Requirements by Epic

### EPIC A: Core Sync Engine
- **A‑1**: Implement feature extractor (frames, windowing, FFT/vDSP, spectral flux, log‑mel/chroma).  
- **A‑2**: Coarse alignment via FFT cross‑correlation (GCC‑PHAT), returning top‑K peaks.  
- **A‑3**: Fine alignment via DTW with band constraint; return best offset and cost.  
- **A‑4**: Drift estimator using windowed offsets → robust line fit (RANSAC); output ppm and keyframes.  
- **A‑5**: Confidence scoring and verification overlay (envelope correlation).  
- **A‑6**: Deterministic parallelization for long files (time‑window partition).  
- **A‑7**: Error handling (`sync_last_error`), input validation, and parameter bounds.  

### EPIC B: Decoding Layer
- **B‑1**: AVFoundation reader: extract mono f32 at target SR.  
- **B‑2**: Format probing and friendly error messages for unsupported containers/codecs.  
- **B‑3**: Optional FFmpeg path (Direct build only) with LGPL-safe linkage; build flag & runtime check.  

### EPIC C: Swift Package & App Glue
- **C‑1**: SPM target compiling the C++ core; expose public header.  
- **C‑2**: Swift wrapper with async `alignFiles(pathA:pathB:)` and event hooks for progress.  
- **C‑3**: Unit tests with small WAV fixtures.  

### EPIC D: macOS App (SwiftUI + SwiftData)
- **D‑1**: Project list, create/delete projects; import clips (NSOpenPanel, bookmarks).  
- **D‑2**: Job queue; run/cancel/resume; show progress & logs.  
- **D‑3**: Result view: offsets, drift, confidence; **alignment heatmap** visualization.  
- **D‑4**: Manual controls: nudge, lock, re‑run local refinement.  
- **D‑5**: Autosave, crash recovery, background processing without UI freeze.  

### EPIC E: Exporters
- **E‑1**: FCPXML generator with offsets and optional warp (time‑map).  
- **E‑2**: Premiere XML exporter with equivalent information.  
- **E‑3**: Resolve CSV/EDL exporter; include documentation for import steps.  
- **E‑4**: CSV/JSON dump of results for audit and scripting.  

### EPIC F: CLI & Tooling
- **F‑1**: `sync_cli` with JSON output and flags (SR, search window, modes).  
- **F‑2**: `verify_dataset.sh` to run CLI over `testdata/` and compare to expected answers (tolerances configurable).  
- **F‑3**: GitHub Actions macOS CI building core, CLI, running unit and dataset tests.  

### EPIC G: Distribution, Licensing, Compliance
- **G‑1**: MAS build with sandbox entitlements; notarization pipeline.  
- **G‑2**: Direct build with Sparkle updates; optional FFmpeg; separate bundle id.  
- **G‑3**: Licensing (subscription) for Direct build with offline grace; privacy policy; no telemetry by default.  

---

## 8. Data Model (SwiftData)

```swift
@Model final class Project {
  @Attribute(.unique) var id: UUID
  var name: String
  var createdAt: Date
  @Relationship(deleteRule: .cascade) var clips: [Clip]
  @Relationship(deleteRule: .cascade) var jobs: [Job]
}

@Model final class Clip {
  @Attribute(.unique) var id: UUID
  var fileURL: URL
  var bookmarkData: Data? // sandbox persistence
  var checksum: String?
  var durationSec: Double?
  var sampleRate: Int?
}

@Model final class Job {
  @Attribute(.unique) var id: UUID
  var startedAt: Date
  var finishedAt: Date?
  var status: String // queued, running, done, failed
  var result: SyncOutcome?
}

@Model final class SyncOutcome {
  var baseOffsetMs: Int64
  var driftPpm: Double
  var confidence: Double
  var keyframeCount: Int
}
```

---

## 9. Acceptance Criteria

- **AC‑1**: Given two overlapping clips with no drift, app reports offset within **±10 ms** (median) and **±20 ms** (P95).  
- **AC‑2**: Given two 60‑minute clips with clock drift (5–30 ppm), app estimates drift within **±2 ppm** (P95) and exports a time‑warp map that keeps alignment error under **±20 ms** across the timeline.  
- **AC‑3**: For non‑overlapping clips, app surfaces a **clear failure** with suggestions (no false success).  
- **AC‑4**: Re‑running the same job produces **identical results** (deterministic).  
- **AC‑5**: Exports import successfully in current versions of **Final Cut Pro**, **Premiere Pro**, and **DaVinci Resolve**.  
- **AC‑6**: MAS build passes App Review (sandboxed, notarized). Direct build updates via Sparkle.  

---

## 10. Risks & Mitigations

- **Format coverage** (AVFoundation limits): Ship Direct build with optional FFmpeg; provide friendly errors & transcode guidance.  
- **Long‑take performance**: Windowed processing; SIMD/vDSP; controlled parallelism; sidecar caching.  
- **Drift edge cases** (temperature or resample artifacts): Robust regression; fallback presets; user nudge + lock.  
- **Licensing complexity**: Start simple (Paddle/Stripe); generous offline grace; document policies.  
- **MAS restrictions**: Security‑scoped bookmarks; avoid writing outside user‑selected folders.  

---

## 11. Milestones (scope‑based, not time‑based)

1. **Walking Skeleton**: Core stubs + Swift wrapper + app shell + CLI prints JSON.  
2. **Feature Pipeline**: Flux + log‑mel/chroma; coarse corr; initial DTW; deterministic runs.  
3. **Drift & Keyframes**: Windowed offsets → robust fit; export warp map; confidence metric.  
4. **UI & Exports**: Heatmap, nudge, FCPXML/Premiere/Resolve, job queue polish.  
5. **Quality Gate**: Golden dataset defined; CI thresholds; perf tuning to target throughput.  
6. **Packaging**: MAS build/entitlements; Direct build with Sparkle; basic licensing.  

---

## 12. Open Questions

- Do we support **Intel Macs** in v1 if performance is below targets? (Default: Apple silicon only.)  
- Minimum OS target: macOS **14** vs **15**? (Default: 14+.)  
- Which **NLE export** nuances need bespoke testing (e.g., audio vs video item offsets, rate conforming)?  
- Do we include a basic **transcription** pass in v1, or hold for v1.1?  

---

## 13. Appendix A — Public C ABI (summary)

```c
// sync_core.h (abridged)
const char* sync_last_error(void);

typedef struct { int sample_rate, hop_ms, window_ms, max_search_minutes, max_keyframes, flags; } sync_params_t;

typedef struct { long long t_src_ms, t_dst_ms; } sync_keyframe_t;

typedef struct { long long base_offset_ms; double drift_ppm, confidence; int keyframe_count; } sync_result_t;

typedef struct sync_session sync_session_t;

sync_session_t* sync_create(const sync_params_t*);
void            sync_destroy(sync_session_t*);
int             sync_feed_a(sync_session_t*, const float*, int);
int             sync_feed_b(sync_session_t*, const float*, int);
int             sync_run(sync_session_t*, sync_result_t*, sync_keyframe_t*, int);
int             sync_align_files(const char* a, const char* b, const sync_params_t*, sync_result_t*, sync_keyframe_t*, int);
```

---

## 14. Appendix B — Repo Layout (reference)

See the separate **Repo Blueprint** document for folders, build files, entitlements, and CI setup. This PRD is the product definition; the blueprint is the implementation scaffold.

