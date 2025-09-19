import Foundation

struct OutputFormatter {
    
    func format(_ result: SyncResult, as format: OutputFormat) throws -> String {
        switch format {
        case .json:
            return try formatAsJSON(result)
        case .csv:
            return formatAsCSV(result)
        case .fcpxml:
            return try formatAsFCPXML(result)
        case .premiere:
            return try formatAsPremiereXML(result)
        case .resolve:
            return formatAsResolveCSV(result)
        }
    }
    
    private func formatAsJSON(_ result: SyncResult) throws -> String {
        let jsonData = try JSONSerialization.data(
            withJSONObject: result.toDictionary(),
            options: .prettyPrinted
        )
        return String(data: jsonData, encoding: .utf8) ?? ""
    }
    
    private func formatAsCSV(_ result: SyncResult) -> String {
        let header = "reference_file,target_file,offset_ms,drift_ppm,confidence,keyframes,processing_time,timestamp"
        let row = [
            result.referenceFile,
            result.targetFile,
            "\(result.offsetMilliseconds)",
            String(format: "%.4f", result.driftPPM),
            String(format: "%.3f", result.confidence),
            "\(result.keyframeCount)",
            String(format: "%.3f", result.processingTimeSeconds),
            ISO8601DateFormatter().string(from: result.timestamp)
        ].joined(separator: ",")
        
        return "\(header)\n\(row)"
    }
    
    private func formatAsFCPXML(_ result: SyncResult) throws -> String {
        let offsetFrames = Int(result.offsetSeconds * 24) // Assume 24fps
        
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.11">
          <resources>
            <format id="r1" name="Main Format" frameDuration="1/24s"/>
            <asset id="r100" name="\(URL(fileURLWithPath: result.referenceFile).lastPathComponent)" src="file://\(result.referenceFile)"/>
            <asset id="r101" name="\(URL(fileURLWithPath: result.targetFile).lastPathComponent)" src="file://\(result.targetFile)"/>
          </resources>
          <library>
            <event name="Sync Results">
              <project name="CLI Sync">
                <sequence format="r1" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                  <spine>
                    <audio-clip ref="r100" offset="0s" name="Reference" start="0s"/>
                    <audio-clip ref="r101" offset="\(String(format: "%.3f", result.offsetSeconds))s" name="Target" start="0s"/>
                  </spine>
                </sequence>
              </project>
            </event>
          </library>
        </fcpxml>
        """
    }
    
    private func formatAsPremiereXML(_ result: SyncResult) throws -> String {
        let timebase = 254016000000
        let offsetTicks = Int64(result.offsetSeconds * Double(timebase))
        
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE xmeml>
        <xmeml version="4">
          <project>
            <name>CLI Sync Project</name>
            <children>
              <sequence id="sequence1">
                <name>Sync Timeline</name>
                <rate>
                  <timebase>\(timebase)</timebase>
                  <ntsc>FALSE</ntsc>
                </rate>
                <media>
                  <audio>
                    <track>
                      <clipitem id="clipitem1">
                        <name>\(URL(fileURLWithPath: result.referenceFile).lastPathComponent)</name>
                        <start>0</start>
                        <file id="file1">
                          <pathurl>file://localhost\(result.referenceFile)</pathurl>
                        </file>
                      </clipitem>
                      <clipitem id="clipitem2">
                        <name>\(URL(fileURLWithPath: result.targetFile).lastPathComponent)</name>
                        <start>\(offsetTicks)</start>
                        <file id="file2">
                          <pathurl>file://localhost\(result.targetFile)</pathurl>
                        </file>
                      </clipitem>
                    </track>
                  </audio>
                </media>
              </sequence>
            </children>
          </project>
        </xmeml>
        """
    }
    
    private func formatAsResolveCSV(_ result: SyncResult) -> String {
        let offsetTC = formatTimecode(seconds: result.offsetSeconds, framerate: 24.0)
        
        let header = "File Name,Media Pool,Start TC,End TC,Track,Enabled,Audio TC,Sync Offset (ms),Confidence"
        let refRow = "\"\(URL(fileURLWithPath: result.referenceFile).lastPathComponent)\",\"Audio\",\"00:00:00:00\",\"\",\"A1\",\"True\",\"00:00:00:00\",\"0\",\"\(String(format: "%.3f", result.confidence))\""
        let targetRow = "\"\(URL(fileURLWithPath: result.targetFile).lastPathComponent)\",\"Audio\",\"\(offsetTC)\",\"\",\"A2\",\"True\",\"\(offsetTC)\",\"\(result.offsetMilliseconds)\",\"\(String(format: "%.3f", result.confidence))\""
        
        return "\(header)\n\(refRow)\n\(targetRow)"
    }
    
    private func formatTimecode(seconds: Double, framerate: Double) -> String {
        let totalFrames = Int(seconds * framerate)
        let hours = totalFrames / Int(framerate * 3600)
        let minutes = (totalFrames % Int(framerate * 3600)) / Int(framerate * 60)
        let secs = (totalFrames % Int(framerate * 60)) / Int(framerate)
        let frames = totalFrames % Int(framerate)
        
        return String(format: "%02d:%02d:%02d:%02d", hours, minutes, secs, frames)
    }
}

extension SyncResult {
    func toDictionary() -> [String: Any] {
        return [
            "schema_version": "1.0",
            "sync_result": [
                "reference_file": referenceFile,
                "target_file": targetFile,
                "offset": [
                    "seconds": offsetSeconds,
                    "milliseconds": offsetMilliseconds
                ],
                "drift": [
                    "ppm": driftPPM,
                    "enabled": true
                ],
                "quality": [
                    "confidence": confidence,
                    "keyframe_count": keyframeCount
                ],
                "processing": [
                    "time_seconds": processingTimeSeconds,
                    "timestamp": ISO8601DateFormatter().string(from: timestamp),
                    "version": version
                ]
            ],
            "metadata": [
                "cli_version": version,
                "generated_at": ISO8601DateFormatter().string(from: Date()),
                "format": "harmoniq_sync_json_v1"
            ]
        ]
    }
}