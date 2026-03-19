# Plan: Extensible Dispatch Table for User-Defined Symbols

## Goal

Allow library consumers to register custom mathematical operations (symbols) without modifying library source code, while preserving all existing performance characteristics for built-in operations.

## Background

The dispatch table is a runtime hash map (keyed by `Node::HashValue`) but its contents are only reachable through compile-time template specializations `Func<T, NodeType, IsContinued, S>` and `Diff<T, NodeType, S>`. A user cannot add a new operation because:

1. `NodeType` is a closed enum — no new values without modifying library headers
2. The dispatch table constructor hardcodes population via `std::make_index_sequence<NodeTypes::Count-3>{}` — only iterates enum values
3. `RegisterCallable` exists but is unusable — it inserts the wrong type into the map (bug)
4. `PrimitiveSet` is tied to `NodeType` via a `Bitset<NodeTypes::Count>` bitmask

## Approach: Option A — Sequential

Build the registration mechanism first (Phases 1–5), framed as the *general* mechanism through which all functions will eventually be registered (not just "custom" extensions). Built-in operations continue to use compile-time specializations throughout, with no performance regression. A subsequent initiative (not in this plan) will migrate built-ins to use the same registration mechanism and collapse `NodeType` to `{Constant, Variable, Function}`.

**Core invariant across all phases:** the built-in hot path (`Func<T, NodeType, IsContinued, S>` specializations, `MakeTuple` constructor, `ForwardPass`) is never touched. All changes add new paths only.

## Phase 1 — Fix `RegisterCallable`

**Files:** `include/operon/core/dispatch.hpp` only

**Problem:** The existing `RegisterCallable` is broken. The map value type is:
```cpp
Tuple = tuple<
    tuple<Callable<T0>, Callable<T1>, ...>,   // TFun
    tuple<CallableDiff<T0>, CallableDiff<T1>, ...>  // TDer
>
```
But the method body does `map_[hash] = make_tuple(f, df)` — inserts the wrong type entirely.

**Changes:** Remove `RegisterCallable`, add a correctly-typed replacement:
```cpp
// Low-level: insert a (Callable, CallableDiff) pair for one scalar type T.
// Creates a new map entry if the hash is absent; updates the type-slot if present.
template<typename T>
requires Operon::Concepts::Arithmetic<T>
void RegisterFunction(Operon::Hash hash, Callable<T> f, CallableDiff<T> df = {});
```

**Test:** Construct `DefaultDispatch`, call `RegisterFunction<Scalar>` with a hand-crafted batch lambda, build a 2-node tree `[Variable(x), Dynamic(myHash, arity=1)]`, evaluate it, verify the result matches the expected scalar computation. Confirm no existing tests regress.

---

## Phase 2 — Scalar-Lambda Adapters

**Files:** new `include/operon/core/symbol_library.hpp`, small additions to `dispatch.hpp`

**Problem:** Users must understand batch kernels (`View<T,S>`, node-index arithmetic, weight multiplication) to use `RegisterFunction`. The API should accept plain scalar lambdas.

**Design note:** This header is named `symbol_library.hpp` (not `custom_symbol.hpp`) and the API uses `RegisterFunction` (not `RegisterCustomSymbol`). The mechanism is designed from the start as the general way to register any function — built-in migration will use the same API in a future phase.

**Adapter signatures (in `symbol_library.hpp`):**
```cpp
// Convert a scalar unary lambda into a batched Callable<T>.
// Handles node weight (nodes[i].Value) multiplication automatically.
template<typename DTable, typename T, typename F>
auto MakeUnaryCallable(F&& primal) -> DTable::template Callable<T>;

// Convert a scalar binary lambda into a batched Callable<T>.
// Handles child index arithmetic (j = i-1, k = j-nodes[j].Length-1).
template<typename DTable, typename T, typename F>
auto MakeBinaryCallable(F&& primal) -> DTable::template Callable<T>;

// Convert a scalar derivative lambda into a batched CallableDiff<T>.
template<typename DTable, typename T, typename DF>
auto MakeUnaryDiff(DF&& deriv) -> DTable::template CallableDiff<T>;

template<typename DTable, typename T, typename DF>
auto MakeBinaryDiff(DF&& deriv) -> DTable::template CallableDiff<T>;
```

**Convenience methods added to `DispatchTable` (in `dispatch.hpp`):**
```cpp
template<typename T, typename F, typename DF = Dispatch::Noop>
void RegisterUnary(Operon::Hash hash, F&& primal, DF&& deriv = {});

template<typename T, typename F, typename DF = Dispatch::Noop>
void RegisterBinary(Operon::Hash hash, F&& primal, DF&& deriv = {});
```

**Critical correctness requirement:** The adapter must apply the node weight (`nodes[parentIndex].Value`) to the result. Built-in `Func<>` specializations do this; a naive adapter that omits it will produce wrong results when a node has a non-unit coefficient after optimization.

**Test:** Register `sin(x) + cos(x)` as a unary function with explicit derivative `cos(x) - sin(x)`. Evaluate a tree containing it, compare against reference. Test again with a non-unit node weight. Test a binary case similarly.

---

## Phase 3 — PrimitiveSet Integration

**Files:** `include/operon/core/pset.hpp` only

**Problem:** `AddPrimitive` already works by hash, but `Node(NodeType::Dynamic, hash)` defaults to arity 0. Users must remember to set `Arity` manually, and there is no guard against the mistake.

**Change:** Add a method that constructs the node correctly:
```cpp
// Register a function symbol for tree generation.
// arity must match what the registered callable expects.
auto AddFunction(Operon::Hash hash, uint16_t arity, size_t frequency = 1) -> bool;
```

**Test:** Add a custom symbol to a `PrimitiveSet` via `AddFunction`. Sample from it in a loop, verify all sampled nodes have the correct arity. Verify that a tree creator can generate trees containing the custom node without assertion failures.

---

## Phase 4 — Auto-Differentiation Fallback

**Files:** `symbol_library.hpp` only (additions), `dispatch.hpp` (small change to `RegisterUnary`/`RegisterBinary`)

**Problem:** When the user provides only a primal lambda, the `CallableDiff` is a `Noop`. This causes `JacRev`/`JacFwd` to silently produce zero/wrong Jacobians through the custom node.

**Approach:** When no explicit derivative is provided, auto-generate a `CallableDiff` using `ceres::Jet<T,1>` (already in the codebase via `dual.hpp`). This is slower than an explicit derivative (one Jet evaluation per batch element in the diff path) but correct. Users who need performance can always supply an explicit derivative.

```cpp
// Auto-differentiating diff adapter using Ceres Jet.
// Requires that F(Jet<T,1>) is valid — i.e. F uses only standard math
// functions that resolve via ADL to Jet overloads.
template<typename DTable, typename T, typename F>
auto MakeUnaryAutoDiff(F&& primal) -> DTable::template CallableDiff<T>;

template<typename DTable, typename T, typename F>
auto MakeBinaryAutoDiff(F&& primal) -> DTable::template CallableDiff<T>;
```

`RegisterUnary`/`RegisterBinary` are updated so that when `DF == Dispatch::Noop`, they substitute `MakeUnary/BinaryAutoDiff` automatically.

**Constraint to document:** The user's lambda must be generic enough to accept `Jet<T,1>` — i.e., it must use `std::` math or other Jet-overloaded functions. Lambdas that branch on the value type or call non-overloaded C functions will not work with auto-diff and must supply an explicit derivative.

**Test:** Register a custom node with primal only (no explicit derivative). Run `JacRev` and `JacFwd` through a tree containing it. Compare against the analytic derivative. Cross-check against the existing autodiff tests. Test that `JacRev` and `JacFwd` agree with each other.

---

## Phase 5 — Node Name Registry

**Files:** `symbol_library.hpp` (additions), `source/core/node.cpp`

**Problem:** `Node::Name()` and `Node::Desc()` look up by `NodeType`, not by hash. All custom nodes share `Type == Dynamic` and display as `"dyn"` in the formatter, serialization output, etc.

**Change:** Add a static per-hash name registry on `Node`:
```cpp
// In node.hpp:
static void RegisterName(Operon::Hash hash, std::string name, std::string desc = {});
[[nodiscard]] auto Name() const noexcept -> std::string const&;  // falls back to "dyn"
[[nodiscard]] auto Desc() const noexcept -> std::string const&;
```

Add a convenience wrapper that does registration in the dispatch table, PrimitiveSet, and name registry in one call:
```cpp
// In symbol_library.hpp:
struct FunctionInfo {
    Operon::Hash hash;
    std::string  name;
    std::string  desc;
    uint16_t     arity;
    size_t       frequency{1};
};

template<typename DTable, typename T, typename F, typename DF = Dispatch::Noop>
void RegisterFunction(DTable& dt, PrimitiveSet& pset,
                      FunctionInfo const& info, F&& primal, DF&& deriv = {});
```

**Test:** Register a custom symbol with a name. Format a tree containing it (infix and postfix). Verify the name appears correctly in formatted output. Verify `Node::Name()` returns the registered name.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Weight multiplication omitted in adapter | Covered explicitly in Phase 2 test with non-unit node weight |
| Auto-diff silently wrong for non-generic lambdas | Document constraint clearly; consider a static_assert or concept check |
| Hash collision between user hash and built-in hash (0–31) | Document: use `Operon::Hasher{}("name")` which produces 64-bit MetroHash; probability of collision with 0–31 is negligible |
| Multi-type dispatch tables (`DispatchTable<float, double>`) | `RegisterFunction<T>` registers one type at a time; document that users must call it per type, or provide an `RegisterAllTypes` helper in Phase 5 |
| `std::function` overhead in custom node path | This is the same cost every node already pays; no regression for built-ins |

## Future Work (not in this plan)

Once this plan is complete and validated, a second initiative can:

1. Replace the dispatch table constructor's index-sequence loop with a call to a `RegisterDefaultLibrary()` function that uses the same `RegisterFunction` API
2. Move arity from implicit enum ordering to explicit `FunctionInfo` metadata
3. Collapse `NodeType` to `{Constant, Variable, Function}`
4. Replace `PrimitiveSetConfig` bitset with a hash-based equivalent
5. Replace infix formatter's 10+ explicit `NodeType` checks with metadata-driven lookup

That work is separable and benefits from having this plan's test coverage already in place.
