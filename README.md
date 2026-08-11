# needle2-rs

Transparent Rust/WASM Needle 2 runtime for official Cactus `.cact` models.
Needle 2 is implemented in the `needle2-format`, `needle2-infer`, and
`needle2-wasm` crates. The existing Needle 1 crates and runtime remain
separate.

## Status

EXPERIMENTAL and TEMPORARY.

## Crates

- `needle2-format` — `.cact` header, tensor directory, tokenizer, and tensor
  decoding.
- `needle2-infer` — Needle 2 math, blocks, mHC/Engram stack, logits, prompt
  rendering, constrained generation, and tool-call validation.
- `needle2-wasm` — transparent browser-facing WASM API.
- `needle-*` — existing Needle 1 implementation; unchanged.

## Native Rust usage

The core API accepts a parsed `CactModel`:

```rust
use needle2_format::CactModel;
use needle2_infer::infer_logits;

let bytes = std::fs::read("needle2.cact")?;
let model = CactModel::parse(&bytes)?;
let tokens = vec![2, 101];
let mut logits = vec![0.0; tokens.len() * needle2_infer::VOCAB_SIZE];
infer_logits(&model, &tokens, &mut logits)?;
```

Tool generation is available through `generate_tool_calls`; it renders the
official tool prompt, performs constrained greedy generation, parses the JSON,
and validates tool names and argument shape.

## WASM build

Install the target and matching binding tool:

```bash
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.121
```

Build browser bindings:

```bash
cargo build -p needle2-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/needle2_wasm.wasm \
  --target web --out-dir dist/needle2-wasm
```

Use the persistent browser handle:

```js
import init, { Needle2Runtime } from "./needle2-wasm/needle2_wasm.js";

await init();
const bytes = new Uint8Array(await (await fetch("needle2.cact")).arrayBuffer());
const runtime = new Needle2Runtime(bytes);
const calls = runtime.complete(JSON.stringify(tools), query, 32);
```

## Tests

Run the complete native workspace suite (excluding the legacy Python and
WASM-host crates):

```bash
cargo test --workspace --exclude needle-wasm --exclude needle-python
```

Run the transparent WASM build checks:

```bash
cargo check -p needle2-wasm --target wasm32-unknown-unknown
cargo build -p needle2-wasm --target wasm32-unknown-unknown --release
```

For real-model tests, set:

```bash
export NEEDLE2_CACT=/path/to/needle2.cact
cargo test -p needle2-format --test cact -- --nocapture
cargo test -p needle2-infer --test block -- --nocapture
cargo test -p needle2-infer --test stack -- --nocapture
```

See [`TESTING.md`](TESTING.md) and [`crates/needle2-wasm/README.md`](crates/needle2-wasm/README.md)
for the compatibility and browser-build details.

## License

MIT for this project. Check the upstream Cactus Compute release for the
license of any model redistributed with it.
