# Needle 2 compatibility testing

The compatibility target is the official `Cactus-Compute/needle2.cact` model and
native `cactus-needle` reference implementation. bside models are not required
for these tests.

## Test matrix

1. **Artifact loading** — valid `.cact` loads; truncated/invalid bytes fail.
2. **Schema initialization** — one tool, multiple tools, OpenAI JSON Schema,
   empty tools, and a null tool-index path.
3. **Reference routing** — compare `function_calls`, arguments, and empty-call
   behavior against the native implementation for fixed query/tool fixtures.
4. **Lifecycle** — completion, reset, reinitialization, and a second model load.
5. **Limits** — long queries, output-capacity exhaustion, and malformed model data.
6. **Browser integration** — execute the same cases through the ESM wrapper in a
   real browser over HTTP (not `file:`).

## Current checks

Rust implementation checks:

```bash
cargo test --workspace --exclude needle-wasm --exclude needle-python
```

Browser smoke check:

```bash
python3 -m http.server 8080
# copy a real model to tests/weights/model.cact
# open tests/browser-smoke.html?model=weights/model.cact
```

The smoke page verifies loading, initialization, valid completion JSON, tool-name
constraints when a call is returned, reset, and a second completion.

## Reference case

With the official native Python package (`cactus-needle 2.0.0`),
`needle2.cact`, and:

```json
[{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
```

this query returns a call:

```text
what's it like in Lagos right now?
```

Expected native result includes:

```json
{"name":"get_weather","arguments":{"city":"Lagos"}}
```

## Current blocker

The same model, tool schema, and query run through the official browser WASM
artifact (`wasm/needle.wasm`) return a valid response with an empty
`function_calls` array and confidence `0.2`. This reproduces when calling the
official Emscripten module directly, bypassing `needle2-rs`, so it is not a
JavaScript wrapper parsing issue.

Until this native-vs-WASM discrepancy is resolved, the project must not claim
full Needle 2 model parity. The wrapper is artifact/load/lifecycle tested, but
semantic routing parity remains failing.
