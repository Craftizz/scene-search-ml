import logging
import asyncio
import torch
from app.services.captioner_service import Captioner, CaptionerConfig
from app.services.embedder_service import Embedder, EmbedderConfig

logger = logging.getLogger("scene_search")


class ModelManager:

    _captioner: Captioner | None = None
    _embedder: Embedder | None = None

    _initialized: bool = False
    _lock: asyncio.Lock = asyncio.Lock()


    @classmethod
    async def initialize(cls) -> None:

        async with cls._lock:
            if cls._initialized:
                logger.warning("Model already initialized")
                return

            try:
                cls._captioner = Captioner(config=CaptionerConfig())
                cls._embedder = Embedder(config=EmbedderConfig())
                cls._initialized = True
                logger.info("ModelManager Models successfully initialized")

            except Exception as e:
                logger.error(f"ModelManager initialization failed: {e}")
                raise


    @classmethod
    def get_captioner(cls) -> Captioner:
        """Get the captioner model instance"""

        if not cls._initialized or cls._captioner is None:
            raise RuntimeError(
                "Model not initialized. Server may still be starting up."
            )

        return cls._captioner
    

    @classmethod
    def get_embedder(cls) -> Embedder:
        """Get the embedder model instance"""

        if not cls._initialized or cls._embedder is None:
            raise RuntimeError(
                "Embedder model not initialized. Server may still be starting up."
            )

        return cls._embedder


    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup on shutdown: move models to CPU, delete references, and clear CUDA cache."""

        async with cls._lock:
            if not cls._initialized:
                return

            try:
                # Captioner cleanup
                if cls._captioner is not None:
                    try:
                        model = getattr(cls._captioner, 'model', None)
                        if model is not None:
                            try:
                                model.to('cpu')
                            except Exception:
                                pass
                            try:
                                delattr(cls._captioner, 'model')
                            except Exception:
                                pass
                    except Exception:
                        logger.exception("Error cleaning up captioner model")

                # Embedder cleanup
                if cls._embedder is not None:
                    try:
                        model = getattr(cls._embedder, 'model', None)
                        if model is not None:
                            try:
                                model.to('cpu')
                            except Exception:
                                pass
                            try:
                                delattr(cls._embedder, 'model')
                            except Exception:
                                pass
                    except Exception:
                        logger.exception("Error cleaning up embedder model")

                # Drop references so GC can reclaim
                cls._captioner = None
                cls._embedder = None

                # Clear CUDA cache if available
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.exception("Error clearing CUDA cache during cleanup")

            finally:
                cls._initialized = False