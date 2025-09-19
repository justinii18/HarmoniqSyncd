import XCTest
@testable import SyncCLI

final class OutputFormatterTests: XCTestCase {
    var formatter: OutputFormatter!
    var testResult: SyncResult!
    
    override func setUp() {
        super.setUp()
        formatter = OutputFormatter()
        testResult = SyncResult(
            referenceFile: "/test/reference.wav",
            targetFile: "/test/target.wav",
            offsetSeconds: 1.25,
            driftPPM: 2.5,
            confidence: 0.89,
            keyframeCount: 42,
            processingTimeSeconds: 3.14,
            timestamp: Date(timeIntervalSince1970: 1640995200), // 2022-01-01 00:00:00 UTC
            version: "1.0.0"
        )
    }
    
    func testJSONFormat() throws {
        let output = try formatter.format(testResult, as: .json)
        
        XCTAssertTrue(output.contains("\"schema_version\": \"1.0\""))
        XCTAssertTrue(output.contains("\"reference_file\": \"/test/reference.wav\""))
        XCTAssertTrue(output.contains("\"target_file\": \"/test/target.wav\""))
        XCTAssertTrue(output.contains("\"seconds\": 1.25"))
        XCTAssertTrue(output.contains("\"milliseconds\": 1250"))
        XCTAssertTrue(output.contains("\"ppm\": 2.5"))
        XCTAssertTrue(output.contains("\"confidence\": 0.89"))
        XCTAssertTrue(output.contains("\"keyframe_count\": 42"))
    }
    
    func testCSVFormat() throws {
        let output = try formatter.format(testResult, as: .csv)
        let lines = output.components(separatedBy: .newlines)
        
        XCTAssertEqual(lines.count, 2)
        XCTAssertTrue(lines[0].contains("reference_file,target_file,offset_ms"))
        XCTAssertTrue(lines[1].contains("/test/reference.wav"))
        XCTAssertTrue(lines[1].contains("/test/target.wav"))
        XCTAssertTrue(lines[1].contains("1250"))
        XCTAssertTrue(lines[1].contains("2.5000"))
        XCTAssertTrue(lines[1].contains("0.890"))
    }
    
    func testFCPXMLFormat() throws {
        let output = try formatter.format(testResult, as: .fcpxml)
        
        XCTAssertTrue(output.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        XCTAssertTrue(output.contains("<!DOCTYPE fcpxml>"))
        XCTAssertTrue(output.contains("<fcpxml version=\"1.11\">"))
        XCTAssertTrue(output.contains("reference.wav"))
        XCTAssertTrue(output.contains("target.wav"))
        XCTAssertTrue(output.contains("offset=\"1.250s\""))
    }
    
    func testPremiereXMLFormat() throws {
        let output = try formatter.format(testResult, as: .premiere)
        
        XCTAssertTrue(output.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        XCTAssertTrue(output.contains("<!DOCTYPE xmeml>"))
        XCTAssertTrue(output.contains("<xmeml version=\"4\">"))
        XCTAssertTrue(output.contains("reference.wav"))
        XCTAssertTrue(output.contains("target.wav"))
    }
    
    func testResolveCSVFormat() throws {
        let output = try formatter.format(testResult, as: .resolve)
        let lines = output.components(separatedBy: .newlines)
        
        XCTAssertEqual(lines.count, 3) // Header + 2 clips
        XCTAssertTrue(lines[0].contains("File Name,Media Pool,Start TC"))
        XCTAssertTrue(lines[1].contains("reference.wav"))
        XCTAssertTrue(lines[1].contains("00:00:00:00")) // Reference at zero
        XCTAssertTrue(lines[2].contains("target.wav"))
        XCTAssertTrue(lines[2].contains("1250")) // Offset in ms
    }
    
    func testTimecodeFormatting() throws {
        let output = try formatter.format(testResult, as: .resolve)
        
        // 1.25 seconds at 24fps = 30 frames = 00:00:01:06
        XCTAssertTrue(output.contains("00:00:01:06"))
    }
}

final class SyncResultTests: XCTestCase {
    
    func testSyncResultDictionary() {
        let result = SyncResult(
            referenceFile: "/test/ref.wav",
            targetFile: "/test/target.wav",
            offsetSeconds: 2.5,
            driftPPM: 1.0,
            confidence: 0.95,
            keyframeCount: 24,
            processingTimeSeconds: 1.5,
            timestamp: Date(timeIntervalSince1970: 1640995200),
            version: "1.0.0"
        )
        
        let dict = result.toDictionary()
        
        XCTAssertEqual(dict["schema_version"] as? String, "1.0")
        
        let syncResult = dict["sync_result"] as? [String: Any]
        XCTAssertNotNil(syncResult)
        XCTAssertEqual(syncResult?["reference_file"] as? String, "/test/ref.wav")
        XCTAssertEqual(syncResult?["target_file"] as? String, "/test/target.wav")
        
        let offset = syncResult?["offset"] as? [String: Any]
        XCTAssertEqual(offset?["seconds"] as? Double, 2.5)
        XCTAssertEqual(offset?["milliseconds"] as? Int, 2500)
        
        let quality = syncResult?["quality"] as? [String: Any]
        XCTAssertEqual(quality?["confidence"] as? Double, 0.95)
        XCTAssertEqual(quality?["keyframe_count"] as? Int, 24)
    }
    
    func testOffsetMillisecondsCalculation() {
        let result = SyncResult(
            referenceFile: "",
            targetFile: "",
            offsetSeconds: 1.234,
            driftPPM: 0,
            confidence: 0,
            keyframeCount: 0,
            processingTimeSeconds: 0,
            timestamp: Date(),
            version: "1.0.0"
        )
        
        XCTAssertEqual(result.offsetMilliseconds, 1234)
    }
}