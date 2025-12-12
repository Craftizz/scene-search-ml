import sendBatch from "./sendBatch";

type Item = {
  file: File;
  resolve: (c: string) => void;
  reject: (e: any) => void;
};

export class BatchUploader {
  private queue: Item[] = [];
  private timer: number | null = null;
  private readonly batchSize: number;
  private readonly batchTimeout: number;
  private readonly apiKey?: string;
  private readonly endpoint?: string;

  constructor(opts?: { batchSize?: number; batchTimeout?: number; apiKey?: string; endpoint?: string }) {
    this.batchSize = opts?.batchSize ?? 4;
    this.batchTimeout = opts?.batchTimeout ?? 1500;
    this.apiKey = opts?.apiKey;
    this.endpoint = opts?.endpoint;
  }

  add(file: File): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      this.queue.push({ file, resolve, reject });
      if (this.queue.length >= this.batchSize) {
        this.flush();
      } else {
        this.scheduleFlush();
      }
    });
  }

  private scheduleFlush() {
    if (this.timer) return;
    this.timer = window.setTimeout(() => {
      this.timer = null;
      this.flush();
    }, this.batchTimeout);
  }

  async flush() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.queue.length === 0) return;

    const items = this.queue.splice(0, this.batchSize);
    const files = items.map((i) => i.file);
    try {
      const captions = await sendBatch(files, this.apiKey, this.endpoint);
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        const cap = captions[i] ?? "";
        it.resolve(cap);
      }
    } catch (e) {
      for (const it of items) it.reject(e);
    }
    // If more items remain, schedule next flush
    if (this.queue.length > 0) this.scheduleFlush();
  }

  dispose() {
    if (this.timer) clearTimeout(this.timer);
    this.queue = [];
  }
}

export default BatchUploader;
