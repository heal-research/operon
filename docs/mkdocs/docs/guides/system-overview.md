# System overview

![Operon subsystems](../assets/diagrams/system-overview.svg){ .diagram }

## Subsystems

The command-line entry points configure a search algorithm around the same core model and evaluator interfaces. Search owns population state and variation; evaluation assigns fitness; execution interprets trees and supplies derivatives or certified bounds; coefficient fitting refines numeric constants.

The interfaces are intentionally layered so a new evaluator or search operator does not need a second tree representation or a parallel execution engine.
