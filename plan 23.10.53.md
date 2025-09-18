# Harmoniq Sync Build Plan

**Current Phase:** Phase 3 – UI Shell & Navigation

## Phase 1 – Repo Foundation
**Status:** Complete
- Scaffold SwiftPM workspace with separate targets for SwiftUI app, C++ core, shared modules
- Check in base CI pipeline (GitHub Actions) running `swift build` + placeholder unit test
- Verify C++ core target compiles with stubbed C ABI
- **Tests:** `swift build`; placeholder XCTest target; C++ stub unit test

## Phase 2 – Data Models & Persistence
**Status:** Complete
- Implement SwiftData models (Project, Clip, Job, SyncOutcome) and persistence helpers
- Add bookmark capture/restore utility for sandboxed file access
- Wire migrations/seeding for sample data where useful
- **Tests:** SwiftData CRUD unit tests; bookmark round-trip test using fixture file

## Phase 3 – UI Shell & Navigation
**Status:** In Progress
- Build SwiftUI navigation (project list, project detail, empty states)
- Establish view models/state containers and dependency injection hooks
- Provide basic theming and accessibility labels
- **Tests:** Snapshot/UI tests covering navigation flow; accessibility lints for primary views

## Phase 4 – Media Import Pipeline
**Status:** Pending
- Implement drag-and-drop + NSOpenPanel import
- Probe metadata (duration, sample rate, channels) and compute checksums
- Persist imported clips/jobs in SwiftData with security-scoped bookmarks
- **Tests:** Integration test importing sample media verifying metadata + checksum consistency

## Phase 5 – Audio Decode Layer
**Status:** Pending
- Build AVFoundation extractor returning mono float32 at target sample rate
- Implement sidecar cache for decoded audio/features
- Add optional FFmpeg decoder hook for Direct build (flagged path)
- **Tests:** Waveform verification against reference audio; performance benchmark ≥3× realtime on base clip

## Phase 6 – Feature Extraction
**Status:** Pending
- Implement spectral flux, log-mel/chroma features with configurable parameters
- Compress/store feature sets for reuse between runs
- Ensure deterministic outputs across machines
- **Tests:** Unit tests comparing features to golden data; determinism check hash

## Phase 7 – Alignment Core
**Status:** Pending
- Implement coarse GCC-PHAT offset search
- Add constrained DTW refinement and drift estimation (RANSAC)
- Finalize stable C ABI for core engine
- **Tests:** Core dataset accuracy tests (offset/drift error thresholds); determinism re-run test

## Phase 8 – Swift Wrapper & Job Orchestration
**Status:** Pending
- Bridge core results into Swift via async job runner and SwiftData updates
- Implement job queue with progress, cancel, resume, and crash recovery hooks
- Persist job state snapshots for resumable sessions
- **Tests:** Integration tests simulating queue operations and recovery from forced crash

## Phase 9 – Explainability UI
**Status:** Pending
- Add confidence indicator, per-clip offsets, and heatmap visualization
- Provide manual nudge controls with re-run capability
- Surface logs/errors with actionable guidance
- **Tests:** UI tests ensuring control interactions update models; snapshot test for heatmap state

## Phase 10 – Exporters
**Status:** Pending
- Implement FCPXML, Premiere XML, Resolve CSV, and keyframe CSV exporters with docs
- Validate exports round-trip into NLEs (headless where possible)
- **Tests:** Schema validation; fixture-based diff tests; scripted NLE import smoke checks

## Phase 11 – CLI & Tooling
**Status:** Pending
- Ship `sync_cli` wrapping core with parity flags and JSON output
- Add dataset verification script for regression testing
- Document CLI usage and scripting hooks
- **Tests:** CLI integration tests on golden dataset; JSON schema contract tests; regression script in CI

## Phase 12 – Performance & Quality Gate
**Status:** Pending
- Profile and optimize hotspots (SIMD, concurrency, caching)
- Finalize golden dataset thresholds and integrate into CI gate
- Harden logging/diagnostics (with zero telemetry by default)
- **Tests:** Automated perf benchmarks hitting ≥3×/5× targets; CI gate enforcing accuracy/perf

## Phase 13 – Packaging & Distribution
**Status:** Pending
- Configure MAS build entitlements + notarization pipeline
- Prepare Direct build with Sparkle updates and licensing stub
- Produce release documentation/checklist
- **Tests:** Notarization dry run; Sparkle update cycle in VM; licensing activation/deactivation test suite
