# Operon — Claude guidance

## External libraries

When working with any external library, consult the official docs first before diving into source code.

- Taskflow: https://taskflow.github.io/

## smol usage

- Use smol only for bounded, source-grounded reconnaissance or summarization to save high-capability-model tokens.
- Verify exact source evidence with repository tools before edits or decisions.
- Escalate design, API, refactor, concurrency, performance, or security work.
- Use TypeSafe only for structured classification after gathering state.
- Avoid long concurrent smol batches because observed throughput is variable.
