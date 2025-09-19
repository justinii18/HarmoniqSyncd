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
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-std=c++17"])
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