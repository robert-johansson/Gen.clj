# Gen.clj MLX: Long-Term Vision

## The Goal

A 100% idiomatic Clojure probabilistic programming system that leverages MLX the way GenJAX leverages JAX — feature-on-par with Gen.jl, with unique advantages that neither Gen.jl nor GenJAX can offer.

---

## Where We Are Today

### Core Gen.clj (main branch)

Five protocol-based abstractions form the complete Generative Function Interface:

- **IGenerativeFunction** — simulate, generate, assess, propose, regenerate
- **ITrace** — execution records with choices, score, update, project
- **IChoiceMap** — hierarchical address → value trees
- **ISelection** — 7 selection types (All, Empty, Set, Hierarchical, Complement, Union, Intersection) + Clojure set extension
- **LogPDF / Sample** — distribution primitives

The dynamic DSL (`gen` macro) rewrites `trace!` and `splice!` calls, threading execution through `*trace*` dynamic var bindings. Three distribution backends (commons-math, java-util, kixi) cover JVM and ClojureScript.

**Inference algorithms (cross-platform):**
- Importance resampling + custom-proposal importance sampling
- Metropolis-Hastings with composable kernels (`mh-step`, `chain`, `cycle-kernels`)
- Particle filtering / SMC with ESS-based resampling

**Combinators (cross-platform):**
- **Map** — apply a kernel to each element of vectorized arguments, with full IUpdate/IProject/IRegenerate
- **Unfold** — sequential state threading for time-series models
- **Switch** — mixture models / conditional branching by index

### MLX Branch (gen-mlx)

Nine source files providing a complete Clojure-to-MLX bridge with gradient-based inference:

| Layer | File | What It Does |
|-------|------|--------------|
| FFI | `mlx/ffi.clj` | Panama/Coffi bindings to mlx-c (650 lines) |
| Array | `mlx/array.clj` | MLXArray implementing IDeref, Seqable, Indexed, IReduce, IMeta, ILookup, IFn |
| Transforms | `mlx/transforms.clj` | grad, value-and-grad, vjp, jvp, compile |
| Distributions | `mlx/distribution.clj` | 5 differentiable distributions: Normal, Exponential, Uniform, Laplace, Cauchy |
| HMC | `mlx/hmc.clj` | Leapfrog integrator + Metropolis-Hastings + MALA |
| MAP | `mlx/map_optimize.clj` | Adam gradient ascent for MAP estimation |
| NUTS | `mlx/nuts.clj` | No-U-Turn Sampler with dual averaging adaptation |
| Dynamic DSL | `mlx/dynamic.clj` | MLX-backed `gen` macro with autodiff traces, HMC/MALA/MAP/NUTS bridges |

The vectorized score function (`build-score-fn`) replays the model once in JVM to collect distribution parameters, then groups by distribution type and performs vectorized MLX logpdf calls — reducing ~6N FFI round-trips to ~6D (D = number of distinct distribution types). Supports all 5 MLX distributions with a fallback path for non-MLX distributions.

A C fast-path shim fuses 8 FFI calls into 1 for compiled closure invocation.

**Test coverage:** 114 MLX tests, 95 CLJ tests, 84 CLJS tests — all passing.

**Benchmarked MLX crossover points:**
- Batched logpdf: MLX faster than kixi at N > 1,000 (1.2x); dominates at N=100K (29x)
- Gradient computation: ~140μs for scalar, ~200μs for N=10K batch
- Scalar operations: MLX loses (~35μs vs ~30ns) due to FFI overhead — expected and acceptable

---

## The Three Pillars

### Pillar 1: Perfect Clojure

Gen.clj should be a showcase of idiomatic Clojure. Not a port that happens to be in Clojure, but a system that could only exist in Clojure — leveraging protocols, immutable data, dynamic binding, REPL-driven development, and cross-platform `.cljc` in ways that feel native.

**Current idiom score: 9/10. Target: 9.5/10.**

Completed improvements:
- ~~Fix core bugs (choicemap.without shadow, can-project?, return type consistency)~~ Done
- ~~Convert `Trace` deftype → defrecord for destructuring, printing, equality~~ Done
- ~~Replace `dotimes` + `volatile!` in `importance/resampling` with `loop/recur`~~ Done

Remaining improvements:
- Extract the ~185-line IFn arity boilerplate into a shared macro
- Protocol-ify distribution parameters (replace hard-coded `(first %)` / `(second %)` with `(dist-params dist)`)
- Resource management: wrap MLX closures in Cleaner pattern (like arrays) instead of global atom keyed by raw pointers
- Consider transducers for the score function vectorization pipeline

### Pillar 2: MLX as the Engine

MLX on Apple Silicon is the compute backend, the way JAX is for GenJAX. Not a thin wrapper — a deep integration where MLX's lazy evaluation, unified memory, autodiff, and compilation are woven into the probabilistic programming abstractions.

**Key insight from GenJAX:** The power comes from `vmap` acting homomorphically on traces — vectorizing both modeling AND inference. Gen.clj's protocol-based GFI is actually better suited for this than Python's decorator pattern. `IGenerativeFunction` can gain a `vectorize` method returning a new generative function on batched inputs, with MLX vmap underneath.

**Where MLX beats JAX for this use case:** MLX's lazy evaluation + unified memory means no CPU-GPU data transfer overhead. For interactive inference (constantly updating traces with new observations), this matters more than raw throughput. Gen.clj + MLX on a MacBook could beat GenJAX on a GPU server for interactive workloads, because the bottleneck is latency.

### Pillar 3: Interactive Inference

This is the unique advantage — something neither Gen.jl nor GenJAX can offer.

GenJAX is purely static: JAX tracing requires fixed computation graphs. Gen.clj's dynamic DSL + Clojure's REPL-driven development enables interactive probabilistic programming. The vision from GEN_ELECTRIC.md: generative functions that interact with the real world, suspending at `trace!` calls and resuming when sensor data arrives.

```clojure
(def agent
  (gen [context]
    (let [h     (trace! :hypothesis prior)
          stim  (trace! :stimulus optimal-design h)
          resp  (trace! :response await-human stim)]
      {:hypothesis h :stimulus stim :response resp})))

;; Particles = competing hypotheses about the human
;; Each particle sees the same real response, resamples based on fit
;; Next stimulus chosen from surviving particles = Bayesian optimal design
```

This requires CPS execution (suspend/resume at trace points), which Clojure can do via Missionary's structured concurrency. JAX cannot.

---

## Phased Roadmap

### Phase 1: Complete the GFI — COMPLETE

*Goal: Full Gen.jl interface on CPU, MLX-accelerated where beneficial.*

**Core GFI completion:**
- [x] Selections — `ISelection` protocol with 7 types + Clojure set extension
- [x] `regenerate` + `project` — `IRegenerate` and `IProject` on all trace types
- [x] MH kernels — `mh-step`, `chain`, `cycle-kernels` (composable)

**MLX distributions:**
- [x] Normal, Exponential, Uniform, Laplace, Cauchy — differentiable logpdfs on MLX
- [x] `IMLXLogPDF` protocol for vectorized batch scoring
- [x] Generalized `build-score-fn` — group-by-type vectorization for mixed-distribution models

**Code quality:**
- [x] Fixed choicemap.without parameter shadow (infinite recursion)
- [x] Fixed `can-project?` using `satisfies?` instead of `instance?`
- [x] Fixed `propose`/`assess` return type inconsistency
- [x] Fixed `propose` destructuring wrong key
- [x] Converted dynamic `Trace` from `deftype` to `defrecord`
- [x] Refactored `importance/resampling` to `loop/recur`

**Verification:**
- [x] Benchmarked scalar/batched logpdf (crossover at N~1K, 29x at N=100K)
- [x] 95 CLJ + 84 CLJS tests passing

### Phase 2: Combinators + Structured Inference — COMPLETE

*Goal: Structured models + sequential Monte Carlo.*

**Combinators:**
- [x] **Map** — `MapGenerativeFn` with full IUpdate/IProject/IRegenerate
- [x] **Unfold** — `UnfoldGenerativeFn` for time series with state threading
- [x] **Switch** — `SwitchGenerativeFn` for mixture models / conditional branching

**Inference:**
- [x] Particle filtering / SMC with ESS-based resampling
- [x] Custom-proposal importance sampling
- [ ] MLX vmap integration for vectorized particle simulation

### Phase 3: Gradient-Based Inference — IN PROGRESS (~40%)

*Goal: Full gradient infrastructure, on par with Gen.jl.*

**Inference algorithms:**
- [x] `IChoiceGradients` for all 5 MLX distributions
- [x] MAP optimization (Adam gradient ascent on MLX)
- [x] MALA (Metropolis-Adjusted Langevin) — HMC with L=1
- [x] NUTS (No-U-Turn Sampler) — adaptive HMC with dual averaging
- [ ] Variational inference (ELBO optimization with MLX autodiff)

**Training:**
- [ ] Trainable parameters with gradient accumulation
- [ ] Parameter learning (EM, wake-sleep)
- [ ] Amortized inference (neural proposal networks)

**MLX depth:**
- [ ] GPU execution via MLX GPU stream
- [ ] Compilation caching — persist compiled kernels across sessions
- [ ] Multivariate Normal with MLX matrix ops + Cholesky
- [ ] Beta, Gamma, Dirichlet, Poisson distributions on MLX

### Phase 4: Advanced Features

*Goal: Feature parity with Gen.jl.*

**Architecture:**
- [ ] Static DSL — compiled trace types with incremental updates (like Gen.jl's `@gen (static)`)
- [ ] Involution MH — generalized reversible-jump MCMC
- [ ] Programmable inference via inference combinators

**Performance:**
- [ ] Arena pooling for hot FFI loops (replace per-call auto-arena)
- [ ] Fused MLX kernels for common inference patterns
- [ ] Memory-mapped arrays for large datasets

**Testing:**
- [ ] Comprehensive statistical validation (Rhat, energy conservation)
- [ ] High-dimensional HMC tests (10D, 100D)
- [ ] Convergence diagnostics built into inference API

### Phase 5: Beyond Gen.jl

*Goal: Capabilities that neither Gen.jl nor GenJAX can offer.*

**Interactive inference:**
- [ ] CPS execution — generative functions that suspend at `trace!` and resume on real-world input
- [ ] Missionary integration — structured concurrent SMC chains
- [ ] Active inference — perception/action/belief unified in single trace

**Electric Clojure integration:**
- [ ] Client/server dissolved — traces span browser + server
- [ ] Real-time inference visualization in browser
- [ ] Collaborative inference — multiple users observing same particle filter

**Cross-platform:**
- [ ] ClojureScript MLX — WebGPU backend for in-browser inference
- [ ] Shared `.cljc` model definitions running on JVM (MLX) and browser (WebGPU)

---

## Where MLX Beats CPU (Benchmarked)

| Workload | CPU Baseline | MLX | Status |
|----------|-------------|-----|--------|
| Batched logpdf (N=1K) | 26μs (kixi) | 21μs (compiled) | 1.2x faster |
| Batched logpdf (N=100K) | 2,160μs (kixi) | 79μs (compiled) | 29x faster |
| Gradient computation | Not available | 140μs (scalar) | Working |
| HMC inference | Not available | Leapfrog + autodiff | Working |
| NUTS inference | Not available | Adaptive tree + dual averaging | Working |
| MAP optimization | Not available | Adam + autodiff | Working |
| Scalar logpdf | 30ns (kixi) | 35μs (MLX) | MLX loses (FFI overhead) |

| Workload | CPU Baseline | MLX Target | Status |
|----------|-------------|------------|--------|
| Vectorized particle filter | Sequential loop | vmap across particles | Not started |
| Multivariate Normal logpdf | Not implemented | MLX matrix ops | Not started |
| Variational inference | Not available | ELBO + MLX autodiff | Not started |
| Static trace updates | Full re-execution | Incremental + compiled | Not started |

MLX will *lose* on scalar operations due to FFI overhead. This is expected and acceptable. The value is in batched, vectorized, and gradient-based workloads.

---

## Architectural Principles

### 1. Dual-path architecture

CPU (kixi/commons-math) for cross-platform + ClojureScript. MLX for Apple Silicon performance. Never force MLX — users opt in via `:mlx` alias. The elegant cross-platform `.cljc` story is preserved while acceleration is available.

### 2. Protocols all the way down

Every extension point is a protocol. New distributions, new inference algorithms, new backends — all added by extending protocols, never by modifying core code. This is Clojure's greatest strength and Gen.clj should lean into it fully.

### 3. Lazy by default

MLX arrays are lazy (computation graph, not values). Traces should be lazy where possible. Inference should be lazy (stream of samples, not eagerly collected vector). This matches both Clojure's philosophy and MLX's execution model.

### 4. REPL-driven inference

Every inference algorithm should be explorable at the REPL. Step through HMC iterations. Inspect particle weights. Visualize trace updates. The REPL is the debugger, the profiler, and the notebook.

### 5. Composition over configuration

Inference programs are composed from small, reusable pieces — not configured via option maps. `(comp mh-step hmc-step resample)` rather than `{:algorithm :smc :kernel :hmc :adaptation :nuts}`.

---

## Comparison with GenJAX and Gen.jl

| Capability | Gen.jl | GenJAX | Gen.clj MLX |
|------------|--------|--------|-------------|
| Language | Julia | Python | Clojure |
| Compute backend | Julia native | JAX (CUDA/TPU) | MLX (Apple Silicon) |
| Autodiff | ReverseDiff.jl | JAX autodiff | MLX autodiff via mlx-c |
| Vectorization | Julia broadcasting | jax.vmap | Planned (MLX vmap) |
| Compilation | Julia JIT | JAX XLA | MLX compile + C shim |
| Static DSL | Yes | Yes (tracing) | Planned (Phase 4) |
| Dynamic DSL | Yes | Yes | Yes |
| Combinators | Map, Unfold, Switch | Map, Unfold, Switch | Yes (Map, Unfold, Switch) |
| Importance sampling | Yes | Yes | Yes (+ custom proposals) |
| MH / MCMC | Yes | Yes | Yes (composable kernels) |
| HMC | Yes | Yes | Yes |
| NUTS | Yes | Yes | Yes (dual averaging) |
| MALA | Yes | — | Yes |
| MAP optimization | Yes | Yes | Yes (Adam) |
| Particle filter / SMC | Yes | Yes | Yes (ESS resampling) |
| Variational inference | Yes | Yes | Planned (Phase 3) |
| Interactive inference | No | No | Planned — CPS + Missionary (Phase 5) |
| Cross-platform | JVM only | Python only | JVM + ClojureScript |
| REPL-driven | Julia REPL | Jupyter | Clojure REPL + Clerk |
| Notebook integration | Pluto.jl | Jupyter | Clerk |
| Unified memory | N/A | No (CPU-GPU transfer) | Yes (MLX unified) |
| Latency-optimized | No | No | Yes (MLX lazy eval) |

Gen.clj's unique positioning: **interactive, low-latency, cross-platform probabilistic programming with REPL-driven development and Apple Silicon acceleration.**

---

## Success Criteria

### Short-term — ACHIEVED
- [x] All core Gen.jl inference algorithms implemented (importance, MH, HMC, SMC)
- [x] MLX demonstrably faster than CPU for batched operations (benchmarked: 29x at N=100K)
- [x] Map/Unfold/Switch combinators working
- [x] Zero known bugs in core GFI
- [x] NUTS and MAP optimization

### Medium-term (next milestone)
- [ ] Variational inference (ELBO + MLX autodiff)
- [ ] Trainable parameters with gradient accumulation
- [ ] MLX vmap for vectorized particle simulation
- [ ] Multivariate Normal with Cholesky
- [ ] Beta, Gamma, Dirichlet, Poisson on MLX
- [ ] Comprehensive benchmark suite comparing CPU, MLX, and (via libpython-clj) JAX

### Long-term
- [ ] Static DSL with incremental trace updates
- [ ] Interactive generative functions with CPS execution
- [ ] Electric Clojure integration for real-time browser-based inference
- [ ] Active inference agents running on Apple Silicon
- [ ] The most beautiful probabilistic programming codebase in any language
