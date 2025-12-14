"use client";

import React, { useRef, useState, useEffect } from "react";

export default function TestPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [scenes, setScenes] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [intervalMs, setIntervalMs] = useState(200);
  const [batchSize, setBatchSize] = useState(8);
  const [batchTimeout, setBatchTimeout] = useState(1.2);
  const [maxPending, setMaxPending] = useState(1000);
  const [apiKey, setApiKey] = useState("");

  function addLog(s: string) {
    setLogs((l) => [new Date().toISOString() + " — " + s, ...l].slice(0, 200));
  }

  useEffect(() => {
    return () => {
      if (ws) ws.close();
      const t = (window as any)._testTimer;
      if (t) clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function connectWs(): Promise<WebSocket> {
    if (ws && ws.readyState === WebSocket.OPEN) return ws;

    const loc = window.location;
    const proto = loc.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams();
    params.set("batch_size", String(batchSize));
    params.set("batch_timeout", String(batchTimeout));
    params.set("max_pending", String(maxPending));
    if (apiKey) params.set("api_key", apiKey);
    // Backend runs on port 8000
    const backendPort = 8000;
    const host = `${loc.hostname}:${backendPort}`;
    const url = `${proto}//${host}/v1/ws/analyze?${params.toString()}`;
    addLog("connecting: " + url);

    return await new Promise((resolve, reject) => {
      try {
        const socket = apiKey ? new WebSocket(url, [apiKey]) : new WebSocket(url);
        socket.binaryType = "arraybuffer";
        socket.onopen = () => {
          addLog("ws open");
          setWs(socket);
          resolve(socket);
        };
        socket.onmessage = (ev) => {
          try {
            const data = typeof ev.data === "string" ? JSON.parse(ev.data) : JSON.parse(new TextDecoder().decode(ev.data as ArrayBuffer));
            // if scene image present, show thumbnail and avoid logging full base64
            if (data && data.type === "scene" && data.image) {
              setScenes((s) => [data.image, ...s].slice(0, 50));
              addLog(`recv scene ts=${data.timestamp}`);
            } else {
              // avoid logging huge image payloads
              if (data && data.image) {
                const copy = { ...data, image: "(image)" };
                addLog("recv: " + JSON.stringify(copy));
              } else {
                addLog("recv: " + JSON.stringify(data));
              }
            }
          } catch (e) {
            addLog("recv non-json message");
          }
        };
        socket.onclose = (ev) => {
          try {
            // @ts-ignore CloseEvent properties
            const code = (ev && (ev as any).code) || (socket as any).closeCode || "unknown";
            const reason = (ev && (ev as any).reason) || "";
            addLog(`ws closed, code=${code} reason=${reason}`);
          } catch (e) {
            addLog("ws closed");
          }
        };
        socket.onerror = (ev) => {
          addLog("ws error");
          // eslint-disable-next-line no-console
          console.error("WebSocket error:", ev, "readyState=", socket.readyState);
        };
        // safety timeout
        setTimeout(() => {
          if (socket.readyState !== WebSocket.OPEN) {
            if (socket.readyState === WebSocket.CLOSED) return reject(new Error("ws closed"));
            // do not reject immediately; leave open for onopen to resolve
          }
        }, 5000);
      } catch (err) {
        reject(err);
      }
    });
  }

  function handleFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    if (videoRef.current) {
      videoRef.current.src = url;
      videoRef.current.load();
    }
  }

  async function startCapture() {
    if (running) return;
    let socket = ws;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      try {
        socket = await connectWs();
      } catch (e) {
        addLog("failed to connect websocket");
        return;
      }
    }
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    if (!video || !canvas) {
      addLog("video or canvas not ready");
      return;
    }
    // ensure video is playing
    try {
      if (video.paused) await video.play();
    } catch (e) {
      addLog("video play failed - user interaction may be required");
    }
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 180;
    const ctx = canvas.getContext("2d")!;
    setRunning(true);
    const timer = setInterval(async () => {
      if (video.paused || video.ended) return;
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob: Blob | null = await new Promise((res) => canvas.toBlob(res as any, "image/jpeg", 0.8));
        if (!blob) return;
        const ts = video.currentTime;
        ws.send(JSON.stringify({ type: "frame_meta", timestamp: ts }));
        const arr = new Uint8Array(await blob.arrayBuffer());
        ws.send(arr.buffer);
        addLog(`sent frame ts=${ts.toFixed(3)}`);
      } catch (e) {
        addLog("capture/send error");
      }
    }, Number(intervalMs) || 200);
    (window as any)._testTimer = timer;
  }

  function stopCapture() {
    const t = (window as any)._testTimer;
    if (t) {
      clearInterval(t);
      (window as any)._testTimer = null;
    }
    setRunning(false);
    addLog("stopped capture");
  }

  return (
    <div style={{ padding: 20, fontFamily: "Inter, Arial, sans-serif" }}>
      <h2>WS Analyze Test</h2>
      <div style={{ marginBottom: 8 }}>
        <label>API Key: </label>
        <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} style={{ width: 300 }} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <label>Batch size: </label>
        <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
        <label style={{ marginLeft: 8 }}>Batch timeout (s): </label>
        <input type="number" step="0.1" value={batchTimeout} onChange={(e) => setBatchTimeout(Number(e.target.value))} />
        <label style={{ marginLeft: 8 }}>Max pending: </label>
        <input type="number" value={maxPending} onChange={(e) => setMaxPending(Number(e.target.value))} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <button onClick={connectWs}>Connect WS</button>
        <button onClick={() => { if (ws) { ws.close(); setWs(null); } }} style={{ marginLeft: 8 }}>Close WS</button>
      </div>

      <div style={{ marginBottom: 8 }}>
        <input type="file" accept="video/*" onChange={handleFile} />
      </div>

      <div>
        <video ref={videoRef} controls style={{ maxWidth: "100%", display: "block", marginBottom: 8 }} />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div style={{ marginTop: 8 }}>
        {!running ? (
          <button onClick={startCapture}>Start Capture</button>
        ) : (
          <button onClick={stopCapture}>Stop Capture</button>
        )}
      </div>

      <div style={{ marginTop: 12 }}>
        <h4>Logs</h4>
        <div style={{ maxHeight: 300, overflow: "auto", background: "#111", color: "#eee", padding: 8 }}>
          {logs.map((l, i) => (
            <div key={i} style={{ fontFamily: "monospace", fontSize: 12 }}>{l}</div>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <h4>Detected Scenes</h4>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {scenes.map((s, i) => (
            <img key={i} src={s} style={{ width: 160, height: 90, objectFit: "cover", border: "1px solid #333" }} />
          ))}
        </div>
      </div>
    </div>
  );
}
