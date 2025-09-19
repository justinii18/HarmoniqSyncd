import Foundation
import AVFoundation
import CryptoKit

struct AudioFileInfo {
    let duration: Double
    let sampleRate: Double
    let channels: Int
    let format: String
    let fileSize: Int64
    let bitDepth: Int?
    let bitRate: Int?
    let checksum: String?
}

class AudioFileAnalyzer {
    
    func analyze(filePath: String) throws -> AudioFileInfo {
        let url = URL(fileURLWithPath: filePath)
        
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw AnalyzerError.fileNotFound(filePath)
        }
        
        // Get file size
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: filePath)
        let fileSize = fileAttributes[.size] as? Int64 ?? 0
        
        // Analyze audio properties
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.fileFormat
        
        let duration = Double(audioFile.length) / format.sampleRate
        let sampleRate = format.sampleRate
        let channels = Int(format.channelCount)
        
        // Determine format string
        let formatString = determineFormatString(for: audioFile)
        
        // Calculate bit depth and bit rate if available
        let bitDepth = format.settings[AVLinearPCMBitDepthKey] as? Int
        let bitRate = calculateBitRate(fileSize: fileSize, duration: duration)
        
        // Calculate checksum for smaller files (< 100MB)
        let checksum = fileSize < 100_000_000 ? try calculateChecksum(filePath: filePath) : nil
        
        return AudioFileInfo(
            duration: duration,
            sampleRate: sampleRate,
            channels: channels,
            format: formatString,
            fileSize: fileSize,
            bitDepth: bitDepth,
            bitRate: bitRate,
            checksum: checksum
        )
    }
    
    private func determineFormatString(for audioFile: AVAudioFile) -> String {
        let format = audioFile.fileFormat
        
        if let formatName = format.settings[AVFormatIDKey] as? UInt32 {
            switch formatName {
            case kAudioFormatLinearPCM:
                return "PCM"
            case kAudioFormatMPEG4AAC:
                return "AAC"
            case kAudioFormatMPEGLayer3:
                return "MP3"
            case kAudioFormatAppleLossless:
                return "ALAC"
            case kAudioFormatFLAC:
                return "FLAC"
            default:
                return "Unknown"
            }
        }
        
        // Fallback to file extension
        let url = audioFile.url
        return url.pathExtension.uppercased()
    }
    
    private func calculateBitRate(fileSize: Int64, duration: Double) -> Int? {
        guard duration > 0 else { return nil }
        return Int((Double(fileSize) * 8) / duration)
    }
    
    private func calculateChecksum(filePath: String) throws -> String {
        let url = URL(fileURLWithPath: filePath)
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
}

enum AnalyzerError: LocalizedError {
    case fileNotFound(String)
    case invalidAudioFile(String)
    case checksumError(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Audio file not found: \(path)"
        case .invalidAudioFile(let path):
            return "Invalid audio file: \(path)"
        case .checksumError(let message):
            return "Checksum calculation failed: \(message)"
        }
    }
}