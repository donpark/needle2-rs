/* tslint:disable */
/* eslint-disable */

export class Needle2Runtime {
    free(): void;
    [Symbol.dispose](): void;
    complete(tools_json: string, query: string, max_new_tokens: number): any;
    constructor(model_bytes: Uint8Array);
    reset(): void;
}

export function needle2_complete(model_bytes: Uint8Array, tools_json: string, query: string, max_new_tokens: number): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_needle2runtime_free: (a: number, b: number) => void;
    readonly needle2_complete: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly needle2runtime_complete: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly needle2runtime_new: (a: number, b: number) => number;
    readonly needle2runtime_reset: (a: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
