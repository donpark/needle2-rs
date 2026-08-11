# needle2-wasm

Transparent Needle 2 WASM bindings.

Build and generate browser bindings:

```bash
cargo build -p needle2-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/needle2_wasm.wasm \
  --target web --out-dir dist/needle2-wasm
```

Browser usage:

```js
import init, { Needle2Runtime } from "./needle2-wasm/needle2_wasm.js";

await init();
const runtime = new Needle2Runtime(await (await fetch("needle2.cact")).arrayBuffer());
const calls = runtime.complete(JSON.stringify(tools), query, 32);
```

The runtime currently prioritizes transparent correctness over speed. It keeps
model bytes resident, but inference still recomputes the full stack for each
generated token until incremental KV caching is implemented.
