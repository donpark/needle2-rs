export type Needle2Options = {
  wasmUrl?: string | URL;
  systemPrompt?: string;
  tools?: unknown[] | string;
  toolIndexPath?: string;
};

export class Needle2Engine {
  static load(cact: ArrayBuffer | Uint8Array, options?: Needle2Options): Promise<Needle2Engine>;
  init(systemPrompt?: string, tools?: unknown[] | string, toolIndexPath?: string): void;
  complete(input: string, maxNewTokens?: number, outputCapacity?: number): Record<string, unknown>;
  reset(): void;
}

export function load(cact: ArrayBuffer | Uint8Array, options?: Needle2Options): Promise<Needle2Engine>;
