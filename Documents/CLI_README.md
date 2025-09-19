# Harmoniq Sync CLI

Command-line interface for professional audio synchronization using the same algorithms as the GUI application.

## Installation

### Build from Source
```bash
git clone https://github.com/your-org/harmoniq-sync
cd harmoniq-sync
swift build -c release
```

The executable will be available at `.build/release/sync_cli`

### System Installation
```bash
# Copy to system path
sudo cp .build/release/sync_cli /usr/local/bin/
```

## Usage

### Basic Synchronization

Synchronize two audio files:
```bash
sync_cli sync reference.wav target.wav
```

With JSON output to file:
```bash
sync_cli sync reference.wav target.wav --format json --output results.json
```

### Export Formats

#### JSON Output (Default)
```bash
sync_cli sync ref.wav target.wav --format json
```

Produces structured JSON with schema validation:
```json
{
  "schema_version": "1.0",
  "sync_result": {
    "reference_file": "ref.wav",
    "target_file": "target.wav",
    "offset": {
      "seconds": 1.25,
      "milliseconds": 1250
    },
    "drift": {
      "ppm": 2.5,
      "enabled": true
    },
    "quality": {
      "confidence": 0.89,
      "keyframe_count": 42
    }
  }
}
```

#### Final Cut Pro XML
```bash
sync_cli sync ref.wav target.wav --format fcpxml --output timeline.fcpxml
```

#### Adobe Premiere XML
```bash
sync_cli sync ref.wav target.wav --format premiere --output project.xml
```

#### DaVinci Resolve CSV
```bash
sync_cli sync ref.wav target.wav --format resolve --output media_list.csv
```

#### Simple CSV
```bash
sync_cli sync ref.wav target.wav --format csv --output results.csv
```

### Batch Processing

Process multiple file pairs from CSV:
```bash
sync_cli batch batch_list.csv --format json --concurrency 4
```

#### Batch File Format
Create a CSV file with reference,target,output columns:
```csv
# Batch synchronization list
reference1.wav,target1.wav,output1.json
reference2.wav,target2.wav,output2.json
reference3.wav,target3.wav,output3.json
```

#### Batch Options
- `--concurrency N`: Process N files simultaneously (default: 4)
- `--continue-on-error`: Don't stop on individual file failures
- `--format FORMAT`: Output format for all files
- `--verbose`: Show detailed progress for each file

### File Information

Analyze audio file properties:
```bash
sync_cli info audio.wav
```

With detailed metadata:
```bash
sync_cli info audio.wav --verbose
```

### Advanced Options

#### Synchronization Parameters
```bash
sync_cli sync ref.wav target.wav \
  --max-offset 120 \
  --fft-size 8192 \
  --enable-drift true \
  --verbose
```

- `--max-offset SECONDS`: Maximum time offset to search (default: 60)
- `--fft-size SIZE`: FFT size for frequency analysis (default: 4096)
- `--enable-drift BOOL`: Enable drift detection (default: true)

#### Output Control
- `--verbose`: Show detailed processing information
- `--quiet`: Suppress progress output (errors only)
- `--output PATH`: Write results to file instead of stdout

## Integration Examples

### Shell Scripting
```bash
#!/bin/bash
for ref in refs/*.wav; do
    target="targets/$(basename "$ref")"
    output="results/$(basename "$ref" .wav).json"
    sync_cli sync "$ref" "$target" --output "$output" --quiet
done
```

### Python Integration
```python
import subprocess
import json

def sync_audio(ref_file, target_file):
    result = subprocess.run([
        'sync_cli', 'sync', ref_file, target_file, '--format', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"Sync failed: {result.stderr}")

# Usage
result = sync_audio('reference.wav', 'target.wav')
offset_ms = result['sync_result']['offset']['milliseconds']
confidence = result['sync_result']['quality']['confidence']
```

### CI/CD Pipeline
```yaml
# GitHub Actions example
- name: Synchronize Audio Files
  run: |
    sync_cli batch audio_pairs.csv --format json --continue-on-error
    
- name: Validate Results
  run: |
    for result in results/*.json; do
      confidence=$(jq '.sync_result.quality.confidence' "$result")
      if (( $(echo "$confidence < 0.8" | bc -l) )); then
        echo "Low confidence in $result: $confidence"
        exit 1
      fi
    done
```

## Output Schema

### JSON Schema (v1.0)
```json
{
  "schema_version": "1.0",
  "sync_result": {
    "reference_file": "string",
    "target_file": "string", 
    "offset": {
      "seconds": "number",
      "milliseconds": "integer"
    },
    "drift": {
      "ppm": "number",
      "enabled": "boolean"
    },
    "quality": {
      "confidence": "number (0-1)",
      "keyframe_count": "integer"
    },
    "processing": {
      "time_seconds": "number",
      "timestamp": "ISO8601 string",
      "version": "string"
    }
  },
  "metadata": {
    "cli_version": "string",
    "generated_at": "ISO8601 string",
    "format": "harmoniq_sync_json_v1"
  }
}
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `64`: Usage error
- `65`: Data format error
- `66`: Cannot open input
- `73`: Cannot create output
- `130`: Interrupted (Ctrl+C)

## Signal Handling

The CLI handles signals gracefully:
- `SIGINT` (Ctrl+C): Graceful shutdown with cleanup
- `SIGTERM`: Graceful shutdown
- `SIGHUP`: Graceful shutdown

Current operations complete before shutdown when possible.

## Performance Notes

- **Memory Usage**: Scales with audio file length and FFT size
- **CPU Usage**: Multi-threaded processing utilizes all cores
- **Disk I/O**: Minimal temporary file usage
- **Batch Processing**: Respects concurrency limits to avoid resource exhaustion

## Troubleshooting

### Common Issues

**"File not found" errors:**
```bash
# Use absolute paths
sync_cli sync /full/path/to/ref.wav /full/path/to/target.wav
```

**Low confidence results:**
```bash
# Try larger FFT size for better frequency resolution
sync_cli sync ref.wav target.wav --fft-size 8192 --verbose
```

**Memory issues with large files:**
```bash
# Reduce batch concurrency
sync_cli batch files.csv --concurrency 2
```

### Debug Mode
```bash
# Maximum verbosity
sync_cli sync ref.wav target.wav --verbose 2>&1 | tee debug.log
```

### Performance Monitoring
```bash
# Time execution
time sync_cli sync ref.wav target.wav

# Memory usage
/usr/bin/time -v sync_cli sync ref.wav target.wav
```

## API Compatibility

The CLI produces identical results to the GUI application, ensuring consistency across workflows. JSON output includes schema versioning for forward compatibility.

## Support

For issues and feature requests:
- GitHub Issues: https://github.com/your-org/harmoniq-sync/issues
- Documentation: https://docs.harmoniq-sync.com
- CLI Reference: `sync_cli --help`