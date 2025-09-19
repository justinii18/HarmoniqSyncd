import XCTest
import ArgumentParser
@testable import SyncCLI

final class CLITests: XCTestCase {
    
    func testSyncCommandArguments() throws {
        // Test basic sync command parsing
        let arguments = ["sync", "/path/to/ref.wav", "/path/to/target.wav", "--format", "json"]
        
        var command = try SyncCommand.parse(arguments)
        
        XCTAssertEqual(command.referenceFile, "/path/to/ref.wav")
        XCTAssertEqual(command.targetFile, "/path/to/target.wav")
        XCTAssertEqual(command.format, .json)
        XCTAssertNil(command.output)
        XCTAssertEqual(command.maxOffset, 60.0)
        XCTAssertFalse(command.verbose)
        XCTAssertFalse(command.quiet)
    }
    
    func testSyncCommandWithAllOptions() throws {
        let arguments = [
            "sync",
            "/ref.wav",
            "/target.wav",
            "--format", "fcpxml",
            "--output", "/output.fcpxml",
            "--max-offset", "120",
            "--fft-size", "8192",
            "--enable-drift", "false",
            "--verbose",
            "--quiet"
        ]
        
        var command = try SyncCommand.parse(arguments)
        
        XCTAssertEqual(command.referenceFile, "/ref.wav")
        XCTAssertEqual(command.targetFile, "/target.wav")
        XCTAssertEqual(command.format, .fcpxml)
        XCTAssertEqual(command.output, "/output.fcpxml")
        XCTAssertEqual(command.maxOffset, 120.0)
        XCTAssertEqual(command.fftSize, 8192)
        XCTAssertFalse(command.enableDrift)
        XCTAssertTrue(command.verbose)
        XCTAssertTrue(command.quiet)
    }
    
    func testBatchCommandArguments() throws {
        let arguments = [
            "batch",
            "/batch.csv",
            "--format", "csv",
            "--concurrency", "8",
            "--continue-on-error",
            "--verbose"
        ]
        
        var command = try BatchCommand.parse(arguments)
        
        XCTAssertEqual(command.batchFile, "/batch.csv")
        XCTAssertEqual(command.format, .csv)
        XCTAssertEqual(command.concurrency, 8)
        XCTAssertTrue(command.continueOnError)
        XCTAssertTrue(command.verbose)
    }
    
    func testInfoCommandArguments() throws {
        let arguments = ["info", "/audio.wav", "--verbose"]
        
        var command = try InfoCommand.parse(arguments)
        
        XCTAssertEqual(command.filePath, "/audio.wav")
        XCTAssertTrue(command.verbose)
    }
    
    func testOutputFormatParsing() throws {
        XCTAssertEqual(OutputFormat(rawValue: "json"), .json)
        XCTAssertEqual(OutputFormat(rawValue: "csv"), .csv)
        XCTAssertEqual(OutputFormat(rawValue: "fcpxml"), .fcpxml)
        XCTAssertEqual(OutputFormat(rawValue: "premiere"), .premiere)
        XCTAssertEqual(OutputFormat(rawValue: "resolve"), .resolve)
        XCTAssertNil(OutputFormat(rawValue: "invalid"))
    }
    
    func testMainCommandConfiguration() {
        let config = SyncCLI.configuration
        
        XCTAssertEqual(config.commandName, "sync_cli")
        XCTAssertEqual(config.abstract, "Professional audio synchronization tool")
        XCTAssertEqual(config.version, "1.0.0")
        XCTAssertEqual(config.subcommands.count, 3)
    }
}