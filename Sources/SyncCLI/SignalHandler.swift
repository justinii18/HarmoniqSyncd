import Foundation
import Dispatch

class SignalHandler {
    private static var isShuttingDown = false
    private static var signalSources: [DispatchSourceSignal] = []
    
    static func setupSignalHandling() {
        // Handle common termination signals
        let signals = [SIGINT, SIGTERM, SIGHUP]
        
        for signal in signals {
            // Ignore the signal in the default way
            Foundation.signal(signal, SIG_IGN)
            
            // Create a dispatch source for the signal
            let signalSource = DispatchSource.makeSignalSource(signal: signal, queue: .main)
            
            signalSource.setEventHandler {
                handleShutdown(signal: signal)
            }
            
            signalSource.resume()
            signalSources.append(signalSource)
        }
    }
    
    private static func handleShutdown(signal: Int32) {
        guard !isShuttingDown else { return }
        isShuttingDown = true
        
        let signalName = signalName(for: signal)
        print("\n\nReceived \(signalName). Shutting down gracefully...")
        
        // Give time for cleanup
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
            print("Forced shutdown after timeout.")
            exit(128 + signal)
        }
        
        // Trigger graceful shutdown
        NotificationCenter.default.post(name: .shutdown, object: nil)
        
        // Exit with appropriate code
        exit(128 + signal)
    }
    
    private static func signalName(for signal: Int32) -> String {
        switch signal {
        case SIGINT:
            return "SIGINT (Ctrl+C)"
        case SIGTERM:
            return "SIGTERM"
        case SIGHUP:
            return "SIGHUP"
        default:
            return "Signal \(signal)"
        }
    }
    
    static var isShutdownRequested: Bool {
        return isShuttingDown
    }
}

extension Notification.Name {
    static let shutdown = Notification.Name("shutdown")
}

// MARK: - Graceful Shutdown Protocol

protocol GracefulShutdown {
    func prepareForShutdown() async
}

class ShutdownManager {
    private var observers: [GracefulShutdown] = []
    
    init() {
        NotificationCenter.default.addObserver(
            forName: .shutdown,
            object: nil,
            queue: nil
        ) { _ in
            Task {
                await self.performGracefulShutdown()
            }
        }
    }
    
    func register(_ observer: GracefulShutdown) {
        observers.append(observer)
    }
    
    private func performGracefulShutdown() async {
        print("Performing graceful shutdown...")
        
        // Give all observers a chance to clean up
        await withTaskGroup(of: Void.self) { group in
            for observer in observers {
                group.addTask {
                    await observer.prepareForShutdown()
                }
            }
        }
        
        print("Graceful shutdown complete.")
    }
}