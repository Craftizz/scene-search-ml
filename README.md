# Scene Search ML

Lightweight app for extracting video frames, generating captions/embeddings, and searching scenes.

## Overview

This repository contains a frontend (Next.js) and a backend (FastAPI) that together let you:

- Upload or drop a video in the browser
- Extract frames from the video (client-side streaming)
- Generate captions and embeddings for frames using backend services
- Search scenes by text similarity and jump to timestamps in the video

## Repository layout

- `backend/` — Python FastAPI application that offers captioning, embedding, and similarity endpoints.
- `frontend/` — Next.js (React) frontend with UI to upload videos, view thumbnails, and search scenes.

## Requirements

- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)

## Development notes

- Frames are managed in a React context: `frontend/src/features/video/context/VideoFramesContext.tsx`.
- Video extraction logic is in `frontend/src/features/video/utils/extractFramesStream.ts`.


