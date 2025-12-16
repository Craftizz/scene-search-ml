export type AnalyzeMessage =
  | { type: "embedded_batch"; timestamps: number[]; embeddings: number[][] }
  | { type: "embedded_chunk"; timestamps: number[]; embeddings: number[][] }
  | { type: "scene"; timestamp: number; image?: string; caption?: string; request_id?: string }
  | { type: "caption_started"; request_id: string }
  | { type: "caption_chunk"; request_id: string; chunk: string; partial?: boolean }
  | { type: "caption_result"; request_id: string; caption: string }
  | { type: "error"; message: string };

export type AnalyzeClientOptions = {
  apiKey?: string;
  batchSize?: number;
  batchTimeout?: number;
  maxPending?: number;
  urlOverride?: string; // e.g. ws://localhost:8000/v1/ws/analyze
};

export class AnalyzeClient {
  private ws: WebSocket | null = null;
  private opts: AnalyzeClientOptions;

  constructor(opts: AnalyzeClientOptions = {}) {
    this.opts = opts;
  }

  async connect(): Promise<WebSocket> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return this.ws;

    const loc = window.location;
    const proto = loc.protocol === "https:" ? "wss:" : "ws:";

    const params = new URLSearchParams();
    params.set(
      "batch_size",
      String(
        this.opts.batchSize ??
          Number(process.env.NEXT_PUBLIC_ANALYZE_BATCH_SIZE || 8)
      )
    );
    params.set(
      "batch_timeout",
      String(
        this.opts.batchTimeout ??
          Number(process.env.NEXT_PUBLIC_ANALYZE_BATCH_TIMEOUT || 1.2)
      )
    );
    params.set(
      "max_pending",
      String(
        this.opts.maxPending ??
          Number(process.env.NEXT_PUBLIC_ANALYZE_MAX_PENDING || 1000)
      )
    );

    const apiKey = this.opts.apiKey ?? process.env.NEXT_PUBLIC_API_KEY ?? "";
    if (apiKey) params.set("api_key", apiKey);

    const url = this.opts.urlOverride
      ? `${this.opts.urlOverride}?${params.toString()}`
      : `${proto}//${loc.hostname}:8000/v1/ws/analyze?${params.toString()}`;

    return await new Promise((resolve, reject) => {
      try {
        const ws = apiKey ? new WebSocket(url, [apiKey]) : new WebSocket(url);
        ws.binaryType = "arraybuffer";
        ws.onopen = () => {
          this.ws = ws;
          resolve(ws);
        };
        ws.onerror = (e) => {
          reject(new Error("analyze websocket error"));
        };
        ws.onclose = () => {
          // allow reconnect next time
          this.ws = null;
        };
      } catch (err) {
        reject(err);
      }
    });
  }

  onMessage(handler: (msg: AnalyzeMessage) => void) {
    const ws = this.ws;
    if (!ws) return;
    ws.onmessage = (ev) => {
      try {
        const data =
          typeof ev.data === "string"
            ? JSON.parse(ev.data)
            : JSON.parse(new TextDecoder().decode(ev.data as ArrayBuffer));
        handler(data as AnalyzeMessage);
      } catch (e) {
        // ignore parse errors
      }
    };
  }

  async sendFrame(timestamp: number, blob: Blob) {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: "frame_meta", timestamp }));
      const arr = new Uint8Array(await blob.arrayBuffer());
      ws.send(arr.buffer);
    } catch {}
  }

  close() {
    const ws = this.ws;
    if (!ws) return;
    try {
      ws.close();
    } catch {}
    this.ws = null;
  }
}

type OnEvent = (msg: any) => void;

let singleton: {
  ws: WebSocket | null;
  ready: Promise<WebSocket> | null;
} | null = null;

function buildUrl(batchParams?: Record<string, string | number>) {
  const loc = window.location;
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  const backendPort = 8000;
  const host = `${loc.hostname}:${backendPort}`;
  const params = new URLSearchParams();
  if (batchParams) {
    for (const k of Object.keys(batchParams))
      params.set(k, String(batchParams[k]));
  }
  return `${proto}//${host}/v1/ws/analyze?${params.toString()}`;
}

export async function connectAnalyzeSocket(
  apiKey?: string,
  batchParams?: Record<string, string | number>
) {
  if (!singleton) singleton = { ws: null, ready: null };
  if (singleton.ws && singleton.ws.readyState === WebSocket.OPEN)
    return singleton.ws;
  if (singleton.ready) return singleton.ready;

  const url = buildUrl(batchParams);
  singleton.ready = new Promise((resolve, reject) => {
    try {
      const socket = apiKey ? new WebSocket(url, [apiKey]) : new WebSocket(url);
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        singleton!.ws = socket;
        singleton!.ready = null;
        resolve(socket);
      };
      socket.onclose = () => {
        singleton = null;
      };
      socket.onerror = (e) => {
        // allow consumers to handle via onmessage/out-of-band; don't reject here once opened
        console.error("Analyze WS error", e);
      };
      // safety timeout
      setTimeout(() => {
        if (
          socket.readyState !== WebSocket.OPEN &&
          singleton &&
          singleton.ready
        ) {
          // if not open yet, reject
          // @ts-ignore
          try {
            reject(new Error("WebSocket connect timeout"));
          } catch {}
        }
      }, 5000);
    } catch (err) {
      singleton = null;
      reject(err);
    }
  });
  return singleton.ready;
}

// Send files over the persistent socket. Collect embedded_batch messages until we have
// at least `expected` embeddings, then resolve. Also forward other messages to onEvent.
export async function sendFilesOverAnalyzeSocket(
  files: File[],
  apiKey?: string,
  options?: { onEvent?: OnEvent; batchParams?: Record<string, string | number> }
) {
  const socket = await connectAnalyzeSocket(apiKey, options?.batchParams);

  return await new Promise<number[][]>(async (resolve, reject) => {
    const collected: number[][] = [];

    const onMessage = (ev: MessageEvent) => {
      try {
        const msg =
          typeof ev.data === "string"
            ? JSON.parse(ev.data)
            : JSON.parse(new TextDecoder().decode(ev.data as ArrayBuffer));
        try {
          // debug incoming message types
          // eslint-disable-next-line no-console
          console.debug("analyzeWsClient.recv", msg?.type);
        } catch {}
        if (options?.onEvent) {
          try {
            options.onEvent(msg);
          } catch (e) {}
        }
        if (
          msg &&
          msg.type === "embedded_batch" &&
          Array.isArray(msg.embeddings)
        ) {
          for (const v of msg.embeddings) collected.push(v as number[]);
          if (collected.length >= files.length) {
            socket.removeEventListener("message", onMessage);
            resolve(collected.slice(0, files.length));
          }
        }
      } catch (err) {
        // ignore non-json
      }
    };

    socket.addEventListener("message", onMessage);

    try {
      // send each file as meta + binary
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const meta = { type: "frame_meta", timestamp: i };
        socket.send(JSON.stringify(meta));
        const arr = new Uint8Array(await f.arrayBuffer());
        socket.send(arr.buffer);
      }
      // do not close the socket; rely on backend batching
    } catch (err) {
      socket.removeEventListener("message", onMessage);
      reject(err);
    }
    // fallback timeout
    setTimeout(() => {
      if (collected.length === 0) {
        socket.removeEventListener("message", onMessage);
        resolve(collected);
      }
    }, 20000);
  });
}

export function closeAnalyzeSocket() {
  if (singleton && singleton.ws) {
    try {
      singleton.ws.close();
    } catch {}
    singleton = null;
  }
}

export default {
  connectAnalyzeSocket,
  sendFilesOverAnalyzeSocket,
  closeAnalyzeSocket,
};
