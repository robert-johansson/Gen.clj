# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gen.clj is a Clojure implementation of the Gen probabilistic programming language (originally from MIT). It provides generative modeling and probabilistic inference on both JVM Clojure and ClojureScript (`.cljc` files for cross-platform code).

**The goal:** A 100% idiomatic Clojure probabilistic programming system that leverages MLX the way GenJAX leverages JAX — feature-on-par with Gen.jl, with unique advantages that neither Gen.jl nor GenJAX can offer: interactive inference via CPS execution, REPL-driven exploration, and cross-platform reach.

The `gen-mlx` branch adds the MLX backend for Apple Silicon acceleration — differentiable distributions, autodiff-enabled traces, and hardware-accelerated inference (HMC, NUTS, MALA, MAP, MH).

## Build & Development Commands

**Prerequisites:** Clojure CLI, Babashka, Node.js (for ClojureScript tests)

```bash
# Run Clojure tests (95 tests)
bb test:clj

# Run ClojureScript tests (84 tests — cross-platform .cljc only, excludes JVM-only backends)
# Requires: npm install (first time, to get shadow-cljs)
bb test:cljs

# Run MLX tests (114 tests — requires Apple Silicon + MLX)
bb test:mlx

# Lint source code (clj-kondo)
bb lint

# Lint dependencies (run once on first setup to populate clj-kondo cache)
bb lint-deps

# Run benchmarks (gamma function benchmarks via criterium, ~2min)
bb benchmark:clj

# Interactive Clerk notebook development (opens localhost:7777)
bb clerk-watch

# Build and install JAR locally
clojure -T:build jar
clojure -T:build install
```

Run a single test namespace:
```bash
clojure -X:test:runner :nses '[gen.dynamic-test]'
```

### Lint Notes

The `gen` macro introduces bindings that clj-kondo cannot resolve, so `bb lint` reports "Unresolved symbol" errors for variables that are `gen` macro arguments (e.g. in examples/ and test/gen/dynamic_test.cljc). These are false positives.

## Architecture

### Core Abstractions (the Generative Function Interface)

Five protocol-based abstractions form the GFI:

1. **IGenerativeFunction** (`gen.generative-function`) — Central protocol. `simulate` executes freely and returns a Trace. `generate` executes with constraints. `IAssess`, `IPropose`, and `IRegenerate` extend it for scoring, proposing, and MCMC reproposal.

2. **ITrace** (`gen.trace`) — Records an execution: args, return value, random choices (as IChoiceMap), log probability score. `IUpdate` extends it for trace updates. `IProject` computes log probability of selected choices.

3. **IChoiceMap** (`gen.choicemap`) — Hierarchical tree mapping addresses to random choices. `Choice` wraps leaf values, `DynamicChoiceMap` and `VectorChoiceMap` handle nesting. Supports `#gen/choice` and `#gen/choicemap` data readers.

4. **Distribution protocols** (`gen.distribution`) — `LogPDF` (log-likelihood) and `Sample` (draw sample). `GenerativeFn` wraps any distribution as a generative function.

5. **ISelection** (`gen.selection`) — Specifies subsets of random choices for MCMC. Types: `AllSelection`, `EmptySelection`, `SetSelection`, `HierarchicalSelection`, `ComplementSelection`, `UnionSelection`, `IntersectionSelection`. Clojure sets work directly as selections: `#{:slope :noise}`.

### Dynamic DSL (`gen.dynamic`)

The `gen` macro defines generative functions. It uses `postwalk` to rewrite `trace!` and `splice!` calls to thread through `*trace*` dynamic var. Different bindings of `*trace*` control execution mode:

```clojure
(def my-model
  (gen [x]
    (let [slope (dynamic/trace! :slope dist/normal 0 1)]
      (+ (* slope x) (dynamic/trace! :noise dist/normal 0 0.1)))))
```

- `trace!` — Address a random choice: `(trace! addr gen-fn & args)`
- `splice!` — Call another gen fn, importing its choices directly (no address nesting)
- `untraced` — Execute code block without tracing

**Execution flow:** `simulate`/`generate`/`assess`/`propose` each bind `*trace*` to a different handler, then call the rewritten function. Each `trace!` call invokes the handler which delegates to the sub-function's corresponding operation and accumulates results.

### Distribution Implementations

Four backends, each extending `LogPDF` and `Sample`:

- **`gen.distribution.commons-math`** — JVM only, wraps Apache Commons Math3 (used in examples/)
- **`gen.distribution.java-util`** — JVM only, wraps `java.util.SplittableRandom`
- **`gen.distribution.kixi`** — Cross-platform via `kixi/stats` (use this for .cljc code)
- **`gen.mlx.distribution`** — JVM only, Apple Silicon. Differentiable logpdf via MLX autodiff. Normal, Exponential, Uniform, Laplace, Cauchy. Implements `IMLXLogPDF` for vectorized batch scoring.

Log-likelihood calculations: `gen.distribution.math.log-likelihood`. Gamma functions: `gen.distribution.math.gamma`.

### Combinators (`gen.combinator.*`)

Three combinators for structured models, all cross-platform (`.cljc`):

- **`gen.combinator.map`** — `MapGenerativeFn` applies a kernel to each element of vectorized arguments. Full IUpdate/IProject/IRegenerate support.
- **`gen.combinator.unfold`** — `UnfoldGenerativeFn` for time-series models with sequential state threading through a kernel function.
- **`gen.combinator.switch`** — `SwitchGenerativeFn` for mixture models / conditional branching by index.

### MLX Backend (`gen.mlx.*`)

Nine files providing a complete Clojure-to-MLX bridge for Apple Silicon acceleration:

| Layer | File | Purpose |
|-------|------|---------|
| FFI | `mlx/ffi.clj` | Panama/Coffi bindings to mlx-c |
| Array | `mlx/array.clj` | MLXArray — IDeref, Seqable, Indexed, IReduce, IMeta, ILookup, IFn |
| Transforms | `mlx/transforms.clj` | grad, value-and-grad, vjp, jvp, compile |
| Distributions | `mlx/distribution.clj` | 5 differentiable distributions + IMLXLogPDF protocol |
| HMC | `mlx/hmc.clj` | Leapfrog integrator + Metropolis-Hastings + MALA |
| MAP | `mlx/map_optimize.clj` | Adam gradient ascent for MAP estimation |
| NUTS | `mlx/nuts.clj` | No-U-Turn Sampler with dual averaging step size adaptation |
| Dynamic DSL | `mlx/dynamic.clj` | MLX-backed `gen` macro with autodiff traces and HMC/MALA/MAP/NUTS bridges |

The `build-score-fn` in `mlx/dynamic.clj` replays the model once in JVM to collect distribution parameters, then performs vectorized MLX logpdf calls grouped by distribution type — reducing ~6N FFI round-trips to ~6D (D = number of distinct distribution types).

### Inference

**Cross-platform (`.cljc`):**
- **`gen.inference.importance`** — Importance resampling + custom-proposal importance sampling. Backend-independent (no kixi coupling).
- **`gen.inference.mh`** — Metropolis-Hastings via `regenerate`. Composable: `mh-step`, `chain` (lazy seq of traces), `cycle-kernels` (compose multiple kernels into one sweep).
- **`gen.inference.particle-filter`** — Sequential Monte Carlo with ESS-based resampling, multinomial resampling, log-ML estimation.

**MLX-accelerated (`.clj`, Apple Silicon):**
- **`gen.mlx.hmc`** — Hamiltonian Monte Carlo with leapfrog integrator + MALA (HMC with L=1).
- **`gen.mlx.nuts`** — No-U-Turn Sampler with recursive tree building, multinomial trajectory sampling, and dual averaging for automatic step size adaptation.
- **`gen.mlx.map-optimize`** — MAP estimation via Adam gradient ascent with convergence detection.
- **`gen.mlx.dynamic`** — Bridges: `hmc-sample`, `mala-sample`, `nuts-sample`, `map-optimize` on MLXTrace.

### Clerk / SCI Integration

- `gen.sci` exposes Gen.clj namespaces (including `gen.selection`, `gen.inference.mh`, `gen.inference.particle-filter`, and all combinators) to SCI for Clerk notebooks
- `gen.clerk.*` provides custom Clerk viewers and callout components
- Example notebooks in `examples/` are the interactive documentation source

## Key Patterns

- **Dynamic var tracing:** `gen.dynamic/*trace*` is rebound per-operation to control how `trace!` behaves. This is the core execution mechanism. Use `active-trace` (not `*trace*` directly) for SCI compatibility.
- **Cross-platform via `.cljc`:** Reader conditionals `#?(:clj ... :cljs ...)` throughout. JVM-only backends use `.clj` extension. Selections, MH, combinators, particle filter, and core GFI are fully cross-platform.
- **Protocol extension:** Distribution backends use `extend-type` to implement `LogPDF`/`Sample` on external library types. `IRegenerate` and `IProject` are also added via `extend-type`.
- **Choicemap as constraints:** Pass a choicemap (or plain map) to `gf/generate` to constrain random choices. Unconstrained addresses are sampled; constrained ones use the provided value. The returned weight is log(p/q).
- **Trace update reversibility:** `trace/update` returns `{:trace :weight :discard}`. Applying the discard as new constraints recovers the original trace — important for Metropolis-Hastings.
- **Selections for MCMC:** `(gf/regenerate trace #{:slope})` re-samples `:slope` from prior, keeps everything else fixed. Used by `mh` kernel.
- **Dual-path architecture:** CPU (kixi/commons-math) for cross-platform + ClojureScript. MLX for Apple Silicon acceleration. Users opt in via `:mlx` alias.
- **Tests use `same/ish`** for floating-point comparison and `test.chuck`/`test.check` for property-based testing.

## What's Been Achieved

### Phase 1 (Complete)

**GFI completion:**
- `ISelection` protocol with 7 types + Clojure set extension
- `IProject` on distribution and dynamic traces
- `IRegenerate` protocol + implementations for primitive and dynamic generative functions
- MH kernel with composable helpers (`mh-step`, `chain`, `cycle-kernels`)

**Bug fixes:**
- `DynamicChoiceMap.without` parameter shadow (infinite recursion)
- `can-project?` using `instance?` instead of `satisfies?`
- `propose`/`assess` return type inconsistency (standardized to maps)
- `propose` destructuring wrong key (`:submap` -> `:choices`)

**Code quality:**
- Dynamic `Trace` converted from `deftype` to `defrecord` (~22 `.-field` accesses replaced)
- `importance/resampling` refactored to `loop/recur`, kixi coupling removed

**MLX distributions:**
- `IMLXLogPDF` protocol for vectorized batch scoring
- 4 new distributions: Exponential, Uniform, Laplace, Cauchy (in addition to Normal)
- `build-score-fn` generalized for mixed-distribution models (group-by-type vectorization)

### Phase 2 (Complete)

**Combinators (cross-platform `.cljc`):**
- **Map** — `MapGenerativeFn` with full IUpdate/IProject/IRegenerate
- **Unfold** — `UnfoldGenerativeFn` for time series with state threading
- **Switch** — `SwitchGenerativeFn` for mixture models / conditional branching

**Inference:**
- Particle filtering / SMC with ESS-based resampling
- Custom-proposal importance sampling

### Phase 3 (In Progress — ~40%)

**Gradient-based inference (complete):**
- NUTS — No-U-Turn Sampler with recursive tree building, dual averaging step size adaptation, multinomial trajectory sampling
- MAP optimization — Adam gradient ascent with convergence detection and early stopping
- MALA — Metropolis-Adjusted Langevin Algorithm (HMC with L=1)
- Bridge functions in `mlx/dynamic.clj`: `nuts-sample`, `map-optimize`, `mala-sample`

**Still to do:**
- Variational inference (ELBO + MLX autodiff)
- Trainable parameters with gradient accumulation
- Multivariate Normal with MLX matrix ops + Cholesky
- Beta, Gamma, Dirichlet, Poisson distributions on MLX

### Pre-Phase 1 (MLX Foundation)

- Complete Clojure-to-MLX FFI bridge via Panama/Coffi
- MLXArray with full Clojure collection protocol support
- Autodiff transforms: grad, value-and-grad, vjp, jvp, compile
- MLX dynamic DSL (`mlx-dyn/gen`) with autodiff-enabled traces
- HMC sampler with leapfrog integrator
- C shim fast-path for compiled closure invocation
- Vectorized score function batching FFI calls

## Roadmap: What's Next

### Phase 3 (Remaining): Gradient-Based Inference

- Variational inference (ELBO + MLX autodiff)
- Trainable parameters with gradient accumulation
- Parameter learning (EM, wake-sleep)
- Multivariate Normal with MLX matrix ops + Cholesky
- Beta, Gamma, Dirichlet, Poisson distributions on MLX
- GPU execution via MLX GPU stream

### Phase 4: Advanced Features

- Static DSL — compiled trace types with incremental updates
- Involution MH — reversible-jump MCMC
- Programmable inference via inference combinators
- MLX vmap for vectorized particle simulation
- Arena pooling for hot FFI loops
- Convergence diagnostics (Rhat) built into inference API

### Phase 5: Interactive Inference (Long-Term Vision)

The unique capability neither Gen.jl nor GenJAX can offer:

- **CPS execution** — generative functions that suspend at `trace!` and resume on real-world input (e.g. human responses in a browser)
- **Missionary integration** — structured concurrent SMC chains with suspension/resumption
- **Electric Clojure** — client/server boundary dissolved; traces span browser + server. Model/inference runs on JVM+MLX, visualization/interaction in browser ClojureScript.
- **Active inference agents** — perception, action, and belief unified in a single trace

The generative function becomes an *interaction protocol* — not a description of how data was generated, but a program that inhabits the world, presents stimuli, waits for responses, and updates beliefs in real-time. Inference IS the interaction.

```clojure
;; A generative function that runs a real experiment:
(def adaptive-experiment
  (gen [context]
    (let [hypothesis (trace! :hypothesis cognitive-prior)
          stimulus   (trace! :stimulus optimal-design hypothesis)
          response   (trace! :response await-participant stimulus)]
      {:hypothesis hypothesis :stimulus stimulus :response response})))

;; 100 particles = 100 competing theories about the participant.
;; Each real response triggers resampling.
;; Stimuli adapt in real-time to maximize information gain.
(particle-filter adaptive-experiment {:n-particles 100})
```

## Benchmarks

Run with `bb test:mlx` (benchmark tests in `gen.mlx.benchmark-inference-test`).

### Hardware

| Spec | Mac Mini M2 | Mac Mini M4 |
|---|---|---|
| CPU Cores | 8 (4P + 4E) | 10 (4P + 6E) |
| GPU Cores | 10 | 10 |
| RAM | 16 GB | 32 GB |

### Inference Benchmarks

**Benchmark 1: Normal Posterior (target μ=3.0, 500 steps)**

| Method | M2 | M4 | Speedup |
|---|---|---|---|
| MH | 17.0 ms | 14.8 ms | 1.1x |
| HMC (L=10, eps=0.1) | 1.54 s | 1.20 s | 1.3x |
| NUTS | 1.10 s | 825 ms | 1.3x |

**Benchmark 2: Linear Regression Scaling (HMC/NUTS wall-clock)**

| N obs | M2 HMC | M4 HMC | M2 NUTS | M4 NUTS |
|---|---|---|---|---|
| 10 | 857 ms | 753 ms | 768 ms | 631 ms |
| 50 | 1.23 s | 1.03 s | 731 ms | 708 ms |
| 100 | 1.47 s | 1.28 s | 1.82 s | 925 ms |

**Benchmark 3: MAP Convergence (target mode=[3,−2])**

| Method | M2 | M4 |
|---|---|---|
| MAP (Adam) | 48.6 ms | 35.2 ms |

**Benchmark 4: Parallel Chains (4 chains x 200 steps, Normal posterior)**

| Method | M2 | M4 |
|---|---|---|
| 4x Sequential | 1.01 s | 722 ms |
| 4x Parallel (vmap) | 469 ms | 498 ms |
| Speedup | 2.1x | 1.4x |

Parallel chains use `vmap` to map a single-sample score function over the batch dimension. The M4's faster single-chain performance leaves less room for parallelism gains.

## Companion Documents

- **VISION.md** — Detailed phased roadmap, architectural principles, comparison with Gen.jl/GenJAX
- **GAPS.md** — Feature gap analysis: 14 gaps identified between Gen.clj and Gen.jl/GenJAX
- **GEN_ELECTRIC.md** — The interactive inference vision: generative functions as interaction protocols, Electric Clojure integration, active inference
- **CLOJURE_REVIEW.md** — Code quality review: idiom issues, functional purity analysis, ecosystem library recommendations
