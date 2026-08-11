import createNeedle from './needle2-runtime.js';

const DEFAULT_OUTPUT_CAPACITY = 64 * 1024;

/** Browser/Node wrapper for Cactus Needle 2's official WASM runtime. */
export class Needle2Engine {
  constructor(module) {
    this.module = module;
  }

  static async load(cact, { wasmUrl, systemPrompt = '', tools = [], toolIndexPath = '' } = {}) {
    const runtimeUrl = wasmUrl ?? new URL('../wasm/needle.wasm', import.meta.url).href;
    const runtimeResponse = await fetch(runtimeUrl);
    if (!runtimeResponse.ok) {
      throw new Error(`Failed to fetch Needle 2 WASM: ${runtimeResponse.status}`);
    }
    const wasmBinary = await runtimeResponse.arrayBuffer();
    const module = await createNeedle({
      wasmBinary,
      noInitialRun: true,
    });
    const bytes = cact instanceof Uint8Array ? cact : new Uint8Array(cact);
    const ptr = module._malloc(bytes.byteLength);
    if (!ptr) throw new Error('Needle 2 model allocation failed');
    try {
      module.HEAPU8.set(bytes, ptr);
      if (Number(module._needle_load(ptr, BigInt(bytes.byteLength))) !== 0) {
        throw new Error('Needle 2 model load failed');
      }
    } finally {
      module._free(ptr);
    }
    const engine = new Needle2Engine(module);
    if (systemPrompt || tools.length || toolIndexPath) {
      engine.init(systemPrompt, tools, toolIndexPath);
    }
    return engine;
  }

  init(systemPrompt = '', tools = [], toolIndexPath = '') {
    const value = this.module.ccall('needle_init', 'number',
      ['string', 'string', 'string'],
      [systemPrompt, typeof tools === 'string' ? tools : JSON.stringify(tools), toolIndexPath]);
    if (Number(value) < 0) throw new Error(`Needle 2 init failed: ${value}`);
  }

  complete(input, maxNewTokens = 256, outputCapacity = DEFAULT_OUTPUT_CAPACITY) {
    const output = this.module._malloc(outputCapacity);
    if (!output) throw new Error('Needle 2 output allocation failed');
    try {
      const length = this.module.ccall('needle_complete', 'number',
        ['string', 'number', 'number', 'number'],
        [input, maxNewTokens, output, outputCapacity]);
      const resultLength = Number(length);
      if (resultLength < 0) throw new Error(`Needle 2 completion failed: ${length}`);
      return JSON.parse(this.module.UTF8ToString(output));
    } finally {
      this.module._free(output);
    }
  }

  reset() {
    this.module.ccall('needle_reset', 'void', [], []);
  }
}

export async function load(cact, options) {
  return Needle2Engine.load(cact, options);
}
