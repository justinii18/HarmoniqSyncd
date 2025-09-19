// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HarmoniqSync",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(
            name: "sync_cli",
            targets: ["SyncCLI"]
        ),
        .library(
            name: "HarmoniqSyncCore",
            targets: ["HarmoniqSyncCore"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.5.0")
    ],
    targets: [
        .executableTarget(
            name: "SyncCLI",
            dependencies: [
                "HarmoniqSyncCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/SyncCLI"
        ),
        .target(
            name: "HarmoniqSyncCore",
            dependencies: [
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/HarmoniqSyncCore",
            sources: [
                "AdvancedGCCPHAT.cpp",
                "AudioDecoder.cpp",
                "AudioFingerprinting.cpp",
                "FFmpegWrapper.cpp",
                "PerformanceOptimizer.cpp",
                "ProductionDTW.cpp",
                "ProductionFFT.cpp",
                "ProductionRANSAC.cpp",
                "sync_core.cpp"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-std=c++17"]),
                .unsafeFlags(["-I/opt/homebrew/include"]), // FFmpeg headers
                .unsafeFlags(["-I/usr/local/include"])     // Alternative FFmpeg location
            ],
            linkerSettings: [
                .linkedLibrary("avformat"),
                .linkedLibrary("avcodec"), 
                .linkedLibrary("avutil"),
                .linkedLibrary("swresample"),
                .linkedLibrary("sqlite3"),                 // SQLite for fingerprint database
                .linkedFramework("Accelerate"),            // Apple's optimized math library
                .linkedFramework("CoreAudioTypes"),
                .unsafeFlags(["-L/opt/homebrew/lib"]),     // FFmpeg libraries
                .unsafeFlags(["-L/usr/local/lib"])         // Alternative FFmpeg location
            ]
        ),
        .testTarget(
            name: "SyncCLITests",
            dependencies: [
                "HarmoniqSyncCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "Tests/SyncCLITests"
        )
    ],
    cxxLanguageStandard: .cxx17
)