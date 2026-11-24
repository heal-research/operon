# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
"""Tiny stdlib-only JSON Schema validator for the documents under docs/schemas.

Implements the draft 2020-12 subset those schemas use: type, enum, const, required, properties,
additionalProperties, items, minItems, maxItems, uniqueItems, minimum, maximum, exclusiveMinimum, minLength,
maxLength, pattern and $ref (to "#/$defs/<name>" or "<file>#/$defs/<name>" inside the schema directory). Any other
validation keyword is rejected so that a schema change cannot silently weaken the smoke tests.

    errors = validate(document, "final.schema.json")   # list of "<path>: <message>", empty when valid
"""

import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"

_ANNOTATIONS = {"$schema", "$comment", "$id", "$defs", "title", "description", "default", "examples"}
_KEYWORDS = {"type", "enum", "const", "required", "properties", "additionalProperties", "items", "minItems", "maxItems",
             "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "minLength", "maxLength", "pattern", "$ref"}
_TYPES = {"null": type(None), "boolean": bool, "integer": int, "number": (int, float), "string": str, "array": list,
          "object": dict}


@lru_cache(maxsize=None)
def _load(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _same(a, b):
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _is(value, name):
    if isinstance(value, bool):
        return name == "boolean"
    return isinstance(value, _TYPES[name])


def _resolve(ref, name):
    target, _, pointer = ref.partition("#")
    node = _load(target or name)
    for part in (p for p in pointer.split("/") if p):
        node = node[part]
    return node, target or name


def _check(value, schema, name, path, errors):
    for key in schema:
        if key not in _ANNOTATIONS and key not in _KEYWORDS and not key.startswith("x-"):
            raise ValueError(f"unsupported schema keyword {key!r} in {name}")
    if "$ref" in schema:
        node, target = _resolve(schema["$ref"], name)
        _check(value, node, target, path, errors)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is(value, t) for t in types):
            errors.append(f"{path}: expected {'|'.join(types)}, got {type(value).__name__}")
            return
    if "const" in schema and not _same(value, schema["const"]):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(_same(value, e) for e in schema["enum"]):
        errors.append(f"{path}: {value!r} not in {schema['enum']!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if len(value) > schema.get("maxLength", len(value)):
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {value!r} does not match {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value!r} not above {schema['exclusiveMinimum']}")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in props:
                _check(child, props[key], name, f"{path}.{key}", errors)
            elif extra is False:
                errors.append(f"{path}.{key}: unexpected property")
            elif isinstance(extra, dict):
                _check(child, extra, name, f"{path}.{key}", errors)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if len(value) > schema.get("maxItems", len(value)):
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and any(_same(a, b) for i, a in enumerate(value) for b in value[:i]):
            errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], name, f"{path}[{i}]", errors)


def validate(document, schema_name):
    """Return the list of violations of docs/schemas/<schema_name> by `document` (empty when valid)."""
    errors = []
    _check(document, _load(schema_name), schema_name, "$", errors)
    return errors
