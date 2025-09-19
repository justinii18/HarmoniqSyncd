import XCTest
import SwiftData
@testable import Harmoniq_Sync

final class ExportTests: XCTestCase {
    var modelContainer: ModelContainer!
    var modelContext: ModelContext!
    var testProject: Project!
    var testResult: SyncOutcome!
    
    override func setUpWithError() throws {
        modelContainer = try ModelContainer(for: Project.self, Clip.self, Job.self, SyncOutcome.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        modelContext = ModelContext(modelContainer)
        
        // Create test data
        testProject = Project(name: "Test Export Project")
        
        let clip1 = Clip(
            fileURL: URL(fileURLWithPath: "/test/audio1.wav"),
            durationSeconds: 120.0,
            sampleRate: 48000,
            channelCount: 2
        )
        
        let clip2 = Clip(
            fileURL: URL(fileURLWithPath: "/test/audio2.wav"),
            durationSeconds: 118.5,
            sampleRate: 48000,
            channelCount: 2
        )
        
        testProject.clips = [clip1, clip2]
        
        testResult = SyncOutcome(
            baseOffsetMs: 1250,
            driftPpm: 2.5,
            confidence: 0.89,
            keyframeCount: 42
        )
        
        let job = Job(status: .done, project: testProject, result: testResult)
        testProject.jobs = [job]
        
        modelContext.insert(testProject)
        modelContext.insert(clip1)
        modelContext.insert(clip2)
        modelContext.insert(testResult)
        modelContext.insert(job)
        
        try modelContext.save()
    }
    
    override func tearDownWithError() throws {
        modelContainer = nil
        modelContext = nil
        testProject = nil
        testResult = nil
    }
    
    // MARK: - FCPXML Export Tests
    
    func testFCPXMLExportSuccess() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test.fcpxml")
        
        try FCPXMLExporter.export(project: testProject, to: tempURL)
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
        
        let content = try String(contentsOf: tempURL)
        XCTAssertTrue(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        XCTAssertTrue(content.contains("<!DOCTYPE fcpxml>"))
        XCTAssertTrue(content.contains("<fcpxml version=\"1.11\">"))
        XCTAssertTrue(content.contains("Test Export Project"))
        XCTAssertTrue(content.contains("audio1.wav"))
        XCTAssertTrue(content.contains("audio2.wav"))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testFCPXMLExportNoResult() throws {
        let projectWithoutResult = Project(name: "No Result Project")
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test.fcpxml")
        
        XCTAssertThrowsError(try FCPXMLExporter.export(project: projectWithoutResult, to: tempURL)) { error in
            XCTAssertTrue(error is FCPXMLExporter.ExportError)
            XCTAssertEqual(error as? FCPXMLExporter.ExportError, .noSyncResult)
        }
    }
    
    func testFCPXMLExportInvalidClipData() throws {
        let projectWithOneClip = Project(name: "Single Clip Project")
        let clip = Clip(fileURL: URL(fileURLWithPath: "/test/audio.wav"))
        projectWithOneClip.clips = [clip]
        
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test.fcpxml")
        
        XCTAssertThrowsError(try FCPXMLExporter.export(project: projectWithOneClip, to: tempURL)) { error in
            XCTAssertTrue(error is FCPXMLExporter.ExportError)
            XCTAssertEqual(error as? FCPXMLExporter.ExportError, .invalidClipData)
        }
    }
    
    // MARK: - Premiere XML Export Tests
    
    func testPremiereXMLExportSuccess() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test.xml")
        
        try PremiereXMLExporter.export(project: testProject, to: tempURL)
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
        
        let content = try String(contentsOf: tempURL)
        XCTAssertTrue(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        XCTAssertTrue(content.contains("<!DOCTYPE xmeml>"))
        XCTAssertTrue(content.contains("<xmeml version=\"4\">"))
        XCTAssertTrue(content.contains("Test Export Project"))
        XCTAssertTrue(content.contains("audio1.wav"))
        XCTAssertTrue(content.contains("audio2.wav"))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    // MARK: - Resolve CSV Export Tests
    
    func testResolveCSVExportSuccess() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test.csv")
        
        try ResolveCSVExporter.export(project: testProject, to: tempURL)
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
        
        let content = try String(contentsOf: tempURL)
        XCTAssertTrue(content.contains("File Name"))
        XCTAssertTrue(content.contains("Media Pool"))
        XCTAssertTrue(content.contains("Start TC"))
        XCTAssertTrue(content.contains("Sync Offset (ms)"))
        XCTAssertTrue(content.contains("audio1.wav"))
        XCTAssertTrue(content.contains("audio2.wav"))
        XCTAssertTrue(content.contains("0.89"))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    // MARK: - Keyframe CSV Export Tests
    
    func testKeyframeCSVExportSuccess() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("keyframes.csv")
        
        try KeyframeCSVExporter.export(project: testProject, to: tempURL)
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path))
        
        let content = try String(contentsOf: tempURL)
        XCTAssertTrue(content.contains("Timestamp (seconds)"))
        XCTAssertTrue(content.contains("Reference Clip"))
        XCTAssertTrue(content.contains("Sync Clip"))
        XCTAssertTrue(content.contains("Offset (ms)"))
        XCTAssertTrue(content.contains("Drift (ppm)"))
        XCTAssertTrue(content.contains("Confidence"))
        XCTAssertTrue(content.contains("audio1.wav"))
        XCTAssertTrue(content.contains("audio2.wav"))
        XCTAssertTrue(content.contains("Generated by Harmoniq Sync"))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testKeyframeCSVExportNoKeyframes() throws {
        let resultWithoutKeyframes = SyncOutcome(
            baseOffsetMs: 1000,
            driftPpm: 1.0,
            confidence: 0.8,
            keyframeCount: 0
        )
        
        let projectWithoutKeyframes = Project(name: "No Keyframes")
        let clip1 = Clip(fileURL: URL(fileURLWithPath: "/test/audio1.wav"))
        let clip2 = Clip(fileURL: URL(fileURLWithPath: "/test/audio2.wav"))
        projectWithoutKeyframes.clips = [clip1, clip2]
        
        let job = Job(status: .done, project: projectWithoutKeyframes, result: resultWithoutKeyframes)
        projectWithoutKeyframes.jobs = [job]
        
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("keyframes.csv")
        
        XCTAssertThrowsError(try KeyframeCSVExporter.export(project: projectWithoutKeyframes, to: tempURL)) { error in
            XCTAssertTrue(error is KeyframeCSVExporter.ExportError)
            XCTAssertEqual(error as? KeyframeCSVExporter.ExportError, .noKeyframeData)
        }
    }
    
    // MARK: - Format Validation Tests
    
    func testFCPXMLStructureValidation() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("structure_test.fcpxml")
        
        try FCPXMLExporter.export(project: testProject, to: tempURL)
        
        let content = try String(contentsOf: tempURL)
        
        // Validate basic XML structure
        XCTAssertTrue(content.hasPrefix("<?xml"))
        XCTAssertTrue(content.contains("<resources>"))
        XCTAssertTrue(content.contains("</resources>"))
        XCTAssertTrue(content.contains("<library>"))
        XCTAssertTrue(content.contains("</library>"))
        XCTAssertTrue(content.contains("<sequence"))
        XCTAssertTrue(content.contains("</sequence>"))
        XCTAssertTrue(content.contains("<spine>"))
        XCTAssertTrue(content.contains("</spine>"))
        XCTAssertTrue(content.hasSuffix("</fcpxml>\n"))
        
        // Validate offset calculation (1250ms = ~30 frames at 24fps)
        XCTAssertTrue(content.contains("offset=\"52.08"))
        
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    func testCSVStructureValidation() throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("csv_test.csv")
        
        try ResolveCSVExporter.export(project: testProject, to: tempURL)
        
        let content = try String(contentsOf: tempURL)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        
        // Should have header + at least 2 data lines
        XCTAssertGreaterThanOrEqual(lines.count, 3)
        
        // Validate CSV header
        let header = lines[0]
        XCTAssertTrue(header.contains("File Name"))
        XCTAssertTrue(header.contains("Sync Offset (ms)"))
        
        // Validate data rows have correct number of columns
        for line in lines.dropFirst() {
            let columns = line.components(separatedBy: ",")
            XCTAssertEqual(columns.count, 9) // Should have 9 columns based on our CSV format
        }
        
        try? FileManager.default.removeItem(at: tempURL)
    }
}