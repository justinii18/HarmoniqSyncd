# Export Formats Documentation

Harmoniq Sync supports exporting synchronization results to multiple industry-standard formats for use in professional video editing applications.

## Supported Export Formats

### Final Cut Pro XML (FCPXML)
**File Extension:** `.fcpxml`  
**Compatible With:** Final Cut Pro X, Final Cut Pro 10.4+

Exports synchronization data as an FCPXML project file containing:
- Audio clips positioned with calculated time offsets
- Proper frame rate and duration metadata
- Timeline structure ready for import

**Usage:**
1. Complete a synchronization job in Harmoniq Sync
2. Click "Final Cut Pro" in the Export section
3. Choose save location and filename
4. Import the `.fcpxml` file directly into Final Cut Pro

### Adobe Premiere XML
**File Extension:** `.xml`  
**Compatible With:** Adobe Premiere Pro, Adobe Media Encoder

Exports synchronization data in Premiere's XML interchange format containing:
- Bin structure with organized clips
- Sequence with precise timecode positioning
- Audio track assignments and metadata

**Usage:**
1. Complete a synchronization job in Harmoniq Sync
2. Click "Adobe Premiere" in the Export section
3. Choose save location and filename
4. Import the `.xml` file via File > Import in Premiere Pro

### DaVinci Resolve CSV
**File Extension:** `.csv`  
**Compatible With:** DaVinci Resolve 17+, Blackmagic Design tools

Exports a CSV file containing:
- File names and paths
- Timecode positions and offsets
- Track assignments
- Confidence ratings for each clip

**CSV Structure:**
```
"File Name","Media Pool","Start TC","End TC","Track","Enabled","Audio TC","Sync Offset (ms)","Confidence"
"audio1.wav","Audio","00:00:00:00","00:02:00:00","A1","True","00:00:00:00","0","0.89"
"audio2.wav","Audio","00:00:52:02","00:01:58:12","A2","True","00:00:52:02","1250","0.89"
```

**Usage:**
1. Complete a synchronization job in Harmoniq Sync
2. Click "DaVinci Resolve" in the Export section
3. Import CSV via Media Pool > Import > Import Media List

### Keyframe CSV Data
**File Extension:** `.csv`  
**Compatible With:** Any application supporting CSV import, custom workflows

Exports detailed keyframe-level synchronization data for analysis or custom processing:
- Timestamp positions throughout the sync analysis
- Per-keyframe offset calculations
- Drift compensation values
- Confidence metrics over time

**CSV Structure:**
```
"Timestamp (seconds)","Reference Clip","Sync Clip","Offset (ms)","Drift (ppm)","Confidence","Notes"
"0.000","audio1.wav","audio2.wav","1250.00","2.5000","0.890","Keyframe 1"
"2.857","audio1.wav","audio2.wav","1250.71","2.5000","0.881","Keyframe 2"
```

## Technical Implementation Details

### Offset Calculation
- Base offset in milliseconds is converted to appropriate time units for each format
- FCPXML: Converted to seconds with frame-accurate positioning
- Premiere XML: Converted to ticks using timebase (254016000000)
- Resolve CSV: Formatted as timecode (HH:MM:SS:FF)

### Drift Compensation
All formats include drift compensation calculated from the RANSAC drift estimation:
- **Parts Per Million (PPM):** Clock speed difference between recording devices
- **Progressive Offset:** Base offset + accumulated drift over time
- **Keyframe Interpolation:** Smooth offset transitions between alignment points

### Quality Metrics
Each export includes confidence ratings derived from:
- Cross-correlation strength during GCC-PHAT analysis
- DTW path consistency
- RANSAC consensus among alignment keyframes
- Signal-to-noise ratio of the source audio

## Validation and Testing

### Export Validation
All exporters include built-in validation:
- Schema compliance checking
- Timecode format validation
- File path verification
- Metadata consistency checks

### Round-Trip Testing
Exported files are tested for compatibility:
- FCPXML: Validated against Apple's FCPXML DTD
- Premiere XML: Tested with Adobe's XML schema
- CSV formats: Verified import into target applications

### Error Handling
Common export errors and solutions:
- **No Sync Result:** Ensure synchronization job completed successfully
- **Invalid Clip Data:** Verify all clips have valid duration and metadata
- **File Write Errors:** Check permissions and disk space
- **Format Validation Errors:** Contact support if persistent

## Best Practices

### File Naming
- Use descriptive names that include project and format
- Avoid special characters in filenames
- Include version numbers for iterative exports

### Workflow Integration
1. **Pre-Export:** Verify sync results and confidence levels
2. **Export Selection:** Choose format based on target NLE
3. **Post-Import:** Check clip positioning and audio levels in target application
4. **Backup:** Keep original sync project files for future reference

### Performance Considerations
- Large projects may take several seconds to export
- CSV exports are fastest, XML exports require more processing
- Network drives may slow export operations

## Troubleshooting

### Common Issues
- **Missing Audio Files:** Ensure original media files are accessible
- **Incorrect Timecodes:** Verify project frame rate settings
- **Import Failures:** Check target application version compatibility

### Support
For export-related issues:
1. Check this documentation first
2. Verify file permissions and paths
3. Test with a smaller sample project
4. Report persistent issues with sample files attached

---

*This documentation covers Harmoniq Sync v1.0 export functionality. Format specifications may be updated in future versions.*