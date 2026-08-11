# Needle 2 WASM runtime

`needle2-wasm` is the transparent WASM entry point for the Rust Needle 2
implementation. It does not wrap the opaque official Cactus WASM artifact.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install -f wasm-bindgen-cli --version 0.2.121
cargo build -p needle2-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/needle2_wasm.wasm \
  --target web --out-dir dist/needle2-wasm
```

## Browser usage

```js
import init, { Needle2Runtime } from "./needle2-wasm/needle2_wasm.js";

await init();
const modelBytes = new Uint8Array(
  await (await fetch("needle2.cact")).arrayBuffer()
);
const runtime = new Needle2Runtime(modelBytes);

const tools = [{
  name: "get_weather",
  description: "Get the weather",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } }
  }
}];

const calls = runtime.complete(
  JSON.stringify(tools),
  "what's it like in Lagos right now?",
  32
);
```

`Needle2Runtime` keeps the model bytes resident. `reset()` is available for
host lifecycle management. Inference currently recomputes the full model for
each generated token, so this API is intended for correctness testing rather
than production latency.

## ABI

The generated module exports:

- `Needle2Runtime(modelBytes)`;
- `Needle2Runtime.complete(toolsJson, query, maxNewTokens)`;
- `Needle2Runtime.reset()`;
- `needle2_complete(modelBytes, toolsJson, query, maxNewTokens)`.

The result is a JavaScript array of validated tool-call objects.

## Compatibility

The model format and tokenizer are read directly from the official `.cact`
artifact. The transparent runtime is independently implemented in Rust. Full
native/browser numerical parity and incremental KV caching are not yet claimed.
