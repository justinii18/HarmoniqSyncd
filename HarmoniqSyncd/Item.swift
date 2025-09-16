//
//  Item.swift
//  HarmoniqSyncd
//
//  Created by Justin Adjei on 17/09/2025.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
