import Foundation
import ArgumentParser
import Logging
import AVFoundation

@available(macOS 10.15, *)
struct SyncCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sync_cli",
        abstract: "Professional audio synchronization tool",
        version: "1.0.0",
        subcommands: [SyncCommand.self, BatchCommand.self, InfoCommand.self],
        defaultSubcommand: SyncCommand.self
    )
}

@available(macOS 10.15, *)
struct SyncCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sync",
        abstract: "Synchronize two audio files"
    )
    
    @Argument(help: "Reference audio file path")
    var referenceFile: String
    
    @Argument(help: "Target audio file path")
    var targetFile: String
    
    @Option(name: .shortAndLong, help: "Output format: json, csv, fcpxml, premiere, resolve")
    var format: OutputFormat = .json
    
    @Option(name: .shortAndLong, help: "Output file path")
    var output: String?
    
    @Option(help: "Maximum offset to search in seconds")
    var maxOffset: Double = 60.0
    
    @Option(help: "FFT size for analysis")
    var fftSize: Int = 4096
    
    @Option(help: "Enable drift estimation")
    var enableDrift: Bool = true
    
    @Flag(name: .shortAndLong, help: "Enable verbose output")
    var verbose: Bool = false
    
    @Flag(name: .shortAndLong, help: "Quiet mode - suppress progress output")
    var quiet: Bool = false
    
    func run() async throws {
        let logger = Logger(label: "sync_cli")
        
        if verbose {
            LoggingSystem.bootstrap { _ in
                var handler = StreamLogHandler.standardOutput(label: "sync_cli")
                handler.logLevel = .debug
                return handler
            }
        } else if quiet {
            LoggingSystem.bootstrap { _ in
                var handler = StreamLogHandler.standardOutput(label: "sync_cli")
                handler.logLevel = .error
                return handler
            }
        }
        
        logger.info("Starting synchronization analysis")
        logger.debug("Reference: \(referenceFile)")
        logger.debug("Target: \(targetFile)")
        
        do {
            let result = try await performSync()
            try await outputResult(result, logger: logger)
        } catch {
            logger.error("Synchronization failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
    
    private func performSync() async throws -> SyncResult {
        let processor = AudioSyncProcessor()
        
        let config = SyncConfiguration(
            maxOffsetSeconds: maxOffset,
            fftSize: UInt32(fftSize),
            enableDriftEstimation: enableDrift,
            verbose: verbose
        )
        
        if !quiet {
            print("Loading audio files...")
        }
        
        return try await processor.synchronize(
            referenceFile: referenceFile,
            targetFile: targetFile,
            configuration: config
        ) { progress in
            if !quiet {
                let percentage = Int(progress * 100)
                print("\rProgress: \(percentage)%", terminator: "")
                fflush(stdout)
            }
        }
    }
    
    private func outputResult(_ result: SyncResult, logger: Logger) async throws {
        if !quiet {
            print("\nSynchronization complete!")
        }
        
        let formatter = OutputFormatter()
        let formattedOutput = try formatter.format(result, as: format)
        
        if let outputPath = output {
            try formattedOutput.write(to: URL(fileURLWithPath: outputPath), atomically: true, encoding: .utf8)
            logger.info("Results written to \(outputPath)")
        } else {
            print(formattedOutput)
        }
    }
}

@available(macOS 10.15, *)
struct BatchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "Batch process multiple file pairs"
    )
    
    @Argument(help: "CSV file with file pairs (reference,target,output)")
    var batchFile: String
    
    @Option(name: .shortAndLong, help: "Output format for all files")
    var format: OutputFormat = .json
    
    @Option(help: "Maximum concurrent operations")
    var concurrency: Int = 4
    
    @Flag(name: .shortAndLong, help: "Continue on errors")
    var continueOnError: Bool = false
    
    @Flag(name: .shortAndLong, help: "Verbose output")
    var verbose: Bool = false
    
    func run() async throws {
        let logger = Logger(label: "sync_cli_batch")
        
        print("Processing batch file: \(batchFile)")
        
        let pairs = try loadBatchPairs()
        let processor = BatchProcessor(concurrency: concurrency, logger: logger)
        
        let results = try await processor.processBatch(
            pairs: pairs,
            format: format,
            continueOnError: continueOnError,
            verbose: verbose
        )
        
        let successCount = results.filter { $0.success }.count
        let totalCount = results.count
        
        print("\nBatch processing complete:")
        print("Success: \(successCount)/\(totalCount)")
        
        if successCount < totalCount {
            let failureCount = totalCount - successCount
            print("Failures: \(failureCount)")
            
            for result in results where !result.success {
                print("  - \(result.pair.reference) + \(result.pair.target): \(result.error ?? "Unknown error")")
            }
            
            if !continueOnError {
                throw ExitCode.failure
            }
        }
    }
    
    private func loadBatchPairs() throws -> [FilePair] {
        let url = URL(fileURLWithPath: batchFile)
        let content = try String(contentsOf: url)
        
        return try content.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map { line in
                let components = line.components(separatedBy: ",")
                guard components.count >= 2 else {
                    throw ValidationError("Invalid line format: \(line)")
                }
                
                return FilePair(
                    reference: components[0].trimmingCharacters(in: .whitespaces),
                    target: components[1].trimmingCharacters(in: .whitespaces),
                    output: components.count > 2 ? components[2].trimmingCharacters(in: .whitespaces) : nil
                )
            }
    }
}

struct InfoCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Display information about audio files or sync results"
    )
    
    @Argument(help: "Audio file path")
    var filePath: String
    
    @Flag(name: .shortAndLong, help: "Show detailed metadata")
    var verbose: Bool = false
    
    func run() throws {
        let analyzer = AudioFileAnalyzer()
        let info = try analyzer.analyze(filePath: filePath)
        
        print("File: \(filePath)")
        print("Duration: \(String(format: "%.2f", info.duration)) seconds")
        print("Sample Rate: \(info.sampleRate) Hz")
        print("Channels: \(info.channels)")
        print("Format: \(info.format)")
        
        if verbose {
            print("File Size: \(info.fileSize) bytes")
            print("Bit Depth: \(info.bitDepth ?? 0) bits")
            print("Bit Rate: \(info.bitRate ?? 0) bps")
            
            if let checksum = info.checksum {
                print("Checksum: \(checksum)")
            }
        }
    }
}

enum OutputFormat: String, CaseIterable, ExpressibleByArgument {
    case json, csv, fcpxml, premiere, resolve
    
    var defaultValueDescription: String {
        switch self {
        case .json: return "JSON format"
        case .csv: return "CSV format"
        case .fcpxml: return "Final Cut Pro XML"
        case .premiere: return "Adobe Premiere XML"
        case .resolve: return "DaVinci Resolve CSV"
        }
    }
}

// Setup signal handling and run CLI
if #available(macOS 10.15, *) {
    SignalHandler.setupSignalHandling()
    await SyncCLI.main()
} else {
    fatalError("macOS 10.15 or later is required")
}