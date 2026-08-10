# Needle 2 WASM wrapper

This fork adds `needle2-rs`, a small JavaScript wrapper around Cactus Compute's
official Needle 2 Emscripten runtime. It is intentionally an adapter, not a
second implementation of the Needle 2 model.

```js
import { Needle2Engine } from "needle2-rs";

const model = await fetch("bside.cact").then(r => r.arrayBuffer());
const engine = await Needle2Engine.load(model, {
  tools: [{
    name: "get_weather",
    description: "Get the weather",
    parameters: { type: "object", properties: { city: { type: "string" } } }
  }]
});

const result = engine.complete("What is the weather in Paris?");
```

The runtime and model are separate: applications provide their own `.cact`
artifact. The WASM runtime is vendored from the official `Cactus-Compute/needle2`
Hugging Face release and is Apache-2.0 licensed.

This target is browser/bundler-first. The official Emscripten loader uses browser
`fetch`; Node consumers should use the official Cactus native package instead.
