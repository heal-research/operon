# Operon record schemas

JSON Schemas (draft 2020-12) for the machine-readable documents the Operon command-line programs
(`operon_gp`, `operon_nsgp`) write. The engine is the producer of these documents, so the schemas live here and are
the producer's contract; a consumer reads them but does not define them. Consumers (for example the opx experiment
runner) vendor a pinned copy and check the `x-version` they were built against.

| Schema file | `schema` value | Version | Written by |
|---|---|---|---|
| `final.schema.json` | `operon.final` | 1.2 | `--final-json PATH` |
| `resolved-config.schema.json` | `operon.resolved_config` | 1.0 | `--print-config` (stdout) and `--config-out PATH` |
| `generation.schema.json` | `operon.generation` | 1.1 | `--generation-log DIR` (one record per line in `generation-NNNNNN.jsonl` segments) |
| `common.schema.json` | (shared `$defs`, no document) | 1.2 | referenced by the schemas above via `common.schema.json#/$defs/...` |

The `schema` and `schema_version` properties of every document name the schema and the version it was written
against; `schema_version` is `"<major>.<minor>"`. The version a schema file describes is its top-level `x-version`
annotation.

## Conventions

* **Null is not zero.** A property that the engine cannot supply is `null`, never a fabricated `0`. Counters that
  are not split into scoring and optimizer parts, a hash of a file that was not written, and similar values are
  therefore nullable in the schema.
* **`producer.unsupported_fields`.** Lists the contract properties the producer could not supply, so that a
  `null` in them is a declared absence rather than a defect (for `operon.final`: `segments` and `record_count` when no
  generation stream is written, could not be started, or could not be sealed).
* **Generation records.** One `operon.generation` record per completed generation, generation 0 (`initialized`)
  included, with a dense per-attempt `seq`. A value the engine cannot count is `null` and is named in
  `validity.unsupported_fields`; metric means cover finite, non-rejected individuals only. With `--final-json`, an
  `operon.final` document for a complete generation stream carries the sealed segments (`segments`, `record_count`,
  byte hashes) and does not list those fields in `producer.unsupported_fields`.
* **Resolved configuration.** `operon.resolved_config` makes every default explicit and lists the dotted paths that
  were defaulted in `defaulted`. An explicitly unlimited budget (`--generations unlimited`, `--evaluations unlimited`)
  is `null`. Unknown extra keys are tolerated at every level so that minor versions can add properties.
* **Non-finite numbers** are never written as bare `NaN`/`Infinity` (invalid JSON); the affected metric is `null`.
* **Hashes** are lowercase hexadecimal SHA-256; `run_id` is 64 hex characters, `attempt_id` a lowercase UUID, both
  passed on the command line with `--run-id` / `--attempt-id`.

## Validating

The schemas use only `type`, `enum`, `const`, `required`, `properties`, `additionalProperties`, `items`, `minItems`,
`maxItems`, `uniqueItems`, `minimum`, `maximum`, `exclusiveMinimum`, `minLength`, `maxLength`, `pattern` and `$ref`
(within `common.schema.json#/$defs` and `#/$defs`), so any draft 2020-12 validator that can resolve relative file
references works. `test/cli/schema_lite.py` is a stdlib-only validator for exactly this subset; the CLI smoke tests
in `test/cli` use it.
