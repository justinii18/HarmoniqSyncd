import Foundation
import Logging

struct FilePair {
    let reference: String
    let target: String
    let output: String?
}

struct BatchResult {
    let pair: FilePair
    let success: Bool
    let syncResult: SyncResult?
    let error: String?
    let processingTime: TimeInterval
}

class BatchProcessor {
    private let concurrency: Int
    private let logger: Logger
    private let semaphore: DispatchSemaphore
    
    init(concurrency: Int = 4, logger: Logger) {
        self.concurrency = concurrency
        self.logger = logger
        self.semaphore = DispatchSemaphore(value: concurrency)
    }
    
    func processBatch(
        pairs: [FilePair],
        format: OutputFormat,
        continueOnError: Bool,
        verbose: Bool
    ) async throws -> [BatchResult] {
        logger.info("Starting batch processing of \(pairs.count) file pairs")
        
        let results = try await withThrowingTaskGroup(of: BatchResult.self) { group in
            var results: [BatchResult] = []
            
            for (index, pair) in pairs.enumerated() {
                group.addTask {
                    await self.processPair(
                        pair: pair,
                        index: index + 1,
                        total: pairs.count,
                        format: format,
                        verbose: verbose
                    )
                }
                
                // Limit concurrency
                if (index + 1) % concurrency == 0 || index == pairs.count - 1 {
                    // Collect results from this batch
                    while let result = try await group.next() {
                        results.append(result)
                        
                        if !result.success && !continueOnError {
                            // Cancel remaining tasks
                            group.cancelAll()
                            throw BatchError.processingFailed(result.error ?? "Unknown error")
                        }
                    }
                }
            }
            
            return results
        }
        
        logger.info("Batch processing completed")
        return results.sorted { $0.pair.reference < $1.pair.reference }
    }
    
    private func processPair(
        pair: FilePair,
        index: Int,
        total: Int,
        format: OutputFormat,
        verbose: Bool
    ) async -> BatchResult {
        let startTime = Date()
        
        await withCheckedContinuation { continuation in
            semaphore.wait()
            continuation.resume()
        }
        
        defer {
            semaphore.signal()
        }
        
        do {
            if verbose {
                print("[\(index)/\(total)] Processing: \(URL(fileURLWithPath: pair.reference).lastPathComponent) + \(URL(fileURLWithPath: pair.target).lastPathComponent)")
            }
            
            let processor = AudioSyncProcessor()
            let config = SyncConfiguration()
            
            let syncResult = try await processor.synchronize(
                referenceFile: pair.reference,
                targetFile: pair.target,
                configuration: config
            ) { progress in
                if verbose {
                    let percentage = Int(progress * 100)
                    print("[\(index)/\(total)] Progress: \(percentage)%")
                }
            }
            
            // Write output if specified
            if let outputPath = pair.output {
                let formatter = OutputFormatter()
                let formattedOutput = try formatter.format(syncResult, as: format)
                try formattedOutput.write(to: URL(fileURLWithPath: outputPath), atomically: true, encoding: .utf8)
            }
            
            let processingTime = Date().timeIntervalSince(startTime)
            
            return BatchResult(
                pair: pair,
                success: true,
                syncResult: syncResult,
                error: nil,
                processingTime: processingTime
            )
            
        } catch {
            logger.error("Failed to process pair \(pair.reference) + \(pair.target): \(error.localizedDescription)")
            
            let processingTime = Date().timeIntervalSince(startTime)
            
            return BatchResult(
                pair: pair,
                success: false,
                syncResult: nil,
                error: error.localizedDescription,
                processingTime: processingTime
            )
        }
    }
}

enum BatchError: LocalizedError {
    case processingFailed(String)
    case invalidBatchFile(String)
    
    var errorDescription: String? {
        switch self {
        case .processingFailed(let message):
            return "Batch processing failed: \(message)"
        case .invalidBatchFile(let message):
            return "Invalid batch file: \(message)"
        }
    }
}