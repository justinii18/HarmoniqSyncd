# Harmoniq Feature Documentation

This document outlines the features implemented in the Harmoniq application, based on the source code and architectural documentation.

## System Architecture Overview

Harmoniq is a Kotlin Multiplatform project with a Jetpack Compose for Desktop UI. It uses a hybrid architecture, combining a Kotlin/JVM frontend for the user interface and application logic with a native C library for performance-critical audio and video processing. The two layers are connected using a Java Native Access (JNA) bridge.

## Core Synchronization Engine

### Audio-Based Sync Algorithm
- **Automatic Audio Waveform Analysis**: The application's native C library, using FFmpeg, analyzes audio from video and audio files.
- **Synchronization without Clapboards or Timecode**: The core synchronization is based on audio fingerprinting, not requiring traditional sync markers.
- **Audio Fingerprinting**: The application generates audio fingerprints for precise alignment. This is done in the native C library, managed by the `AudioFingerprintService` in the Kotlin layer.
- **Drift Correction**: The synchronization engine can detect and correct for audio drift in long clips. This is a key feature of the native C library.

### Performance
- **Background Processing**: Synchronization is performed in the background, allowing the UI to remain responsive. This is managed by Kotlin Coroutines.
- **Progress Reporting**: The application provides detailed progress updates during the synchronization process, including the current operation and percentage completion. This is achieved through a callback from the native C library to the Kotlin layer.

## File Management & Media Support

### Supported Formats
- **Video Formats**: The application supports common video formats like MP4, MOV, AVI, MKV, and WEBM.
- **Audio Formats**: The application supports common audio formats like MP3, WAV, AAC, FLAC, and M4A.
- **Professional Formats**: The application has some support for professional formats like R3D, MXF, ProRes, DNxHD, and BRAW, primarily for file identification.

### Import & Organization
- **Drag and Drop Media Import**: Users can add files to the application through the Jetpack Compose UI.
- **Automatic Media Detection**: The application automatically determines the file type (video, audio, or unknown).
- **Track-Based Organization**: Files are organized into "tracks" in the `MainViewModel`.
- **Batch Import**: Users can import all media files from multiple folders at once.
- **Smart File Organization**: The application can group files by camera or source during batch import.

### Project Management
- **Reference Track Selection**: Users can designate a specific file as the "reference" for synchronization. The `MainViewModel` ensures that there is always one reference track.
- **File/Track Management**: The `MainViewModel` provides functions to remove tracks, rename them, and reorder them.
- **Non-Destructive Workflow**: The original media files are never modified.

## Export & NLE Integration

### Target NLE Support
- **Final Cut Pro X**: The primary export target is Final Cut Pro X.
- **FCPXML Export**: The application's native C library generates a `.fcpxml` file that can be imported into Final Cut Pro X.

### Export Options
- **User-Selectable Frame Rate**: The user can choose the frame rate for the exported sequence.
- **Export Location**: The user can select where to save the exported file.
- **Open in Final Cut Pro**: After exporting, the user can choose to open the file directly in Final Cut Pro.

## User Interface & Experience

### Main Interface
- **File List**: The main interface, built with Jetpack Compose, displays a list of tracks and the files within them.
- **Processing Status**: The `MainViewModel` manages the application's state, which is reflected in the UI to show the current processing status (e.g., Idle, Processing, Success, Error).
- **Real-time Sync Preview**: The UI provides real-time updates on the synchronization process.
- **Color Coding**: The application uses a color-coding system to indicate the status of files.

### Workflow Features
- **One-Click Sync**: A single button in the UI initiates the synchronization process.
- **Manual Sync Adjustment**: Users can influence the sync by selecting the reference track.

### Error Handling & Feedback
- **Clear Error Messages**: The `MainViewModel` provides error messages to the UI for failed synchronizations.
- **Processing Time Estimates**: The progress bar gives an indication of the time remaining.

## Advanced Features

### Professional Workflow
- **Multicam Angle Identification**: The application can group tracks into multicam "groups".
- **Automatic Multicam Setup**: The application can automatically detect and suggest multicam setups from the imported files.

### Quality Control
- **Sync Confidence Indicators**: The native C library calculates a confidence score for each synchronized file, which is displayed in the UI.
- **Waveform Visualization**: The application can generate and display waveform data for audio files, which can be used for manual verification of the sync.

## Project Persistence

- **Save/Load Project**: The application can save the current state of a project (including all files, tracks, and sync data) to a file, and load it back later. This is handled by the `MainViewModel` and the native C library.