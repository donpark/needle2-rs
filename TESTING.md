# Needle 2 testing

The transparent Needle 2 implementation is tested independently of bside
against the official `Cactus-Compute/needle2.cact` artifact and the public
Cactus Python implementation where reference values are available.

## Full suite

```bash
cargo test --workspace --exclude needle-wasm --exclude needle-python
```

This covers the existing Needle 1 runtime plus the new Needle 2 crates. A
non-fatal unreachable-code warning in the legacy `needle-core` quantization
module is currently expected.

## Real model

```bash
export NEEDLE2_CACT=/path/to/needle2.cact
cargo test -p needle2-format --test cact -- --nocapture
cargo test -p needle2-infer --test block -- --nocapture
cargo test -p needle2-infer --test stack -- --nocapture
```

These tests exercise the actual official model rather than a mock fixture.
They verify `.cact` loading, tokenizer IDs, tensor decoding, a complete block,
the 27-layer stack, mHC lane mixing, Engram loading/injection, and logits.

## Reference coverage

Reference fixtures cover:

- embedded tokenizer IDs;
- FP16/FP32 conversion;
- CQ tensor values;
- RMSNorm;
- RoPE;
- Hadamard transform and MLP;
- Engram hashing;
- attention-block outputs;
- prompt rendering;
- JSON/tool-call parsing and validation.

The full stack currently has execution coverage on the official artifact. Full
native hidden-state/logit parity for every mHC and Engram path remains a future
numerical-validation task; do not describe the project as native-compatible
without that comparison.

## WASM

Install the target and matching binding tool:

```bash
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.121
```

Check and build:

```bash
cargo check -p needle2-wasm --target wasm32-unknown-unknown
cargo build -p needle2-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/needle2_wasm.wasm \
  --target web --out-dir dist/needle2-wasm
```

The generated browser API is `Needle2Runtime`, which owns the model bytes and
exposes `complete(toolsJson, query, maxNewTokens)` and `reset()`.

## Browser smoke test

Serve the repository over HTTP. The existing legacy browser smoke pages test
the official adapter and are not tests of the new transparent WASM ABI yet.
For the transparent ABI, use the generated bindings and a real `.cact` model:

```js
import init, { Needle2Runtime } from "./needle2-wasm/needle2_wasm.js";
await init();
const model = new Uint8Array(await (await fetch("needle2.cact")).arrayBuffer());
const runtime = new Needle2Runtime(model);
const calls = runtime.complete(JSON.stringify(tools), query, 32);
```

## Known limitations

- Scalar CQ decoding and full-token recomputation make inference slow.
- The browser ABI is built and bindgen-tested, but a committed browser
  end-to-end test is still needed.
- Confidence pooling primitives and confidence-head tensor loading exist, but
  full hidden-cell collection is not wired into the public completion result.
- Token-level constrained decoding currently validates JSON structure and tool
  schemas; it is not a complete JSON-Schema grammar compiler.
