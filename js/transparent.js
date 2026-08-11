import init, { Needle2Runtime as WasmNeedle2Runtime } from '../wasm/transparent/needle2_wasm.js';

export async function loadNeedle2Runtime(modelBytes) {
  await init();
  return new WasmNeedle2Runtime(modelBytes);
}

export { WasmNeedle2Runtime as Needle2Runtime };
