export type ToolCall = Record<string, unknown>;

export class Needle2Runtime {
  constructor(modelBytes: Uint8Array);
  complete(toolsJson: string, query: string, maxNewTokens: number): ToolCall[];
  reset(): void;
}

export function loadNeedle2Runtime(modelBytes: Uint8Array): Promise<Needle2Runtime>;
