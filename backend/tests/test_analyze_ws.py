import sys
import os
from pathlib import Path
import io
from types import SimpleNamespace

import pytest

# ensure backend 'app' package is importable when tests run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from starlette.testclient import TestClient
from PIL import Image

from app.core import config as config_module
from app.api.v1.ws import analyze as analyze_module


class DummyEmbedder:
    async def embed_images(self, imgs):
        # return a simple vector per image
        out = []
        for i, _ in enumerate(imgs):
            out.append(SimpleNamespace(vector=[float(i)] * 16))
        return out


def make_jpeg_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (2, 2), (255, 0, 0))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    # monkeypatch embedder provider
    monkeypatch.setattr(analyze_module.ModelManager, "get_embedder", staticmethod(lambda: DummyEmbedder()))

    # force immediate flush by lowering max_batch_size
    monkeypatch.setattr(config_module.settings, "max_batch_size", 1)

    # make detection always report a scene at index 0
    monkeypatch.setattr(analyze_module, "detect_boundaries", lambda embeddings_np, prev_embedding=None: [{"index": 0}])

    app = FastAPI()
    app.include_router(analyze_module.router)
    with TestClient(app) as client:
        yield client


def test_ws_analyze_flow(client):
    path = "/v1/ws/analyze"
    with client.websocket_connect(path) as ws:
        # send one frame (meta + bytes)
        ws.send_json({"type": "frame_meta", "timestamp": 0.5})
        ws.send_bytes(make_jpeg_bytes())

        # first message should be embedded_batch
        msg = ws.receive_json()
        assert msg["type"] == "embedded_batch"
        assert isinstance(msg.get("timestamps"), list)
        assert isinstance(msg.get("embeddings"), list)

        # then embedded_chunk
        msg2 = ws.receive_json()
        assert msg2["type"] == "embedded_chunk"

        # then scene meta
        msg3 = ws.receive_json()
        assert msg3["type"] == "scene"
