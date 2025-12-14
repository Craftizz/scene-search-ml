import { sendFilesOverAnalyzeSocket } from "./analyzeWebsocketClient";

type SendEmbeddingOptions = {
  onEvent?: (msg: any) => void;
  batch_size?: number;
};

export default async function sendEmbeddingBatch(files: File[], apiKey?: string, options?: SendEmbeddingOptions) {
  // Delegate to persistent analyze WS client to avoid opening a new socket per batch.
  const batchParams: Record<string, string | number> = {};
  if (options?.batch_size) batchParams["batch_size"] = options.batch_size;
  const embeddings = await sendFilesOverAnalyzeSocket(files, apiKey, { onEvent: options?.onEvent, batchParams });
  return embeddings;
}
