# needle2-rs

A small ESM JavaScript wrapper around Cactus Compute's official Needle 2
WASM runtime. It loads a Needle 2 `.cact` model and exposes tool routing in
browsers and other ESM environments with browser-compatible `fetch`.

This repository is a deployment adapter. It does not reimplement the model,
tokenizer, quantization, or training pipeline.

## API

```js
import { Needle2Engine } from "needle2-rs";

const model = new Uint8Array(
  await fetch("/weights/bside.cact").then(response => response.arrayBuffer())
);

const engine = await Needle2Engine.load(model, {
  wasmUrl: "/wasm/needle.wasm",
});

engine.init("", [{
  name: "get_weather",
  description: "Get the weather",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
  },
}]);

const result = engine.complete("What is the weather in Paris?");
// { function_calls: [{ name: "get_weather", arguments: {...} }] }
```

`Needle2Engine.load()` accepts an `ArrayBuffer` or `Uint8Array`. The model and
WASM runtime are separate assets. `wasmUrl` must resolve to the runtime file;
applications should use their asset URL mechanism (for example,
`chrome.runtime.getURL()` in an MV3 extension).

The wrapper supports the official runtime's `.cact` format, including embedded
tokenizer metadata and mixed quantization. It does not accept the older
SafeTensors + `vocab.txt` format.

## Runtime support

The shipped Emscripten loader is browser/bundler-first and uses `fetch` to load
WASM. Node consumers should provide a browser-compatible fetchable URL or use
the official Cactus native package. A local `file:` URL is not a supported WASM
loading path.

## Verification

The Rust implementation has unit, functional, constrained-decoder, FFI, and
reference-vector tests:

```bash
cargo test --workspace --exclude needle-wasm --exclude needle-python
```

Tests that need model weights are skipped unless the corresponding reference
weights are available. The checked-in reference-vector tests do not require a
model download.

The JavaScript wrapper has a browser smoke page at
`tests/browser-smoke.html`. Serve the repository over HTTP, copy a real `.cact`
model into `tests/weights/`, and open:

```text
http://localhost:8080/tests/browser-smoke.html?model=weights/model.cact
```

The smoke test verifies runtime loading, `.cact` loading, initialization,
JSON completion, tool-name validity, reset, and a second completion. It uses
no mock model.

## Model and runtime provenance

- Model architecture, training, `.cact` format, and official runtime: Cactus
  Compute Needle 2.
- This project: JavaScript/ESM adapter and browser asset loading.

See `docs/needle2-wasm.md` for the browser integration notes.

## License

MIT for this adapter. Check the upstream Cactus Compute release for the
license of any model or runtime artifact you redistribute.
