import logging
from app.services.captioner_service import Captioner, CaptionerConfig
from app.services.embedder_service import Embedder, EmbedderConfig

logger = logging.getLogger("scene_search")

class ModelManager:

    _captioner: Captioner | None = None
    _embedder: Embedder | None = None

    _initialized: bool = False


    @classmethod
    async def initialize(cls) -> None:

        if cls._initialized:
            logger.warning("Model already initialized")
            return

        try:
            cls._captioner = Captioner(config=CaptionerConfig())
            cls._embedder = Embedder(config=EmbedderConfig())
            cls._initialized = True
            logger.info("Model successfully initialized")

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
        """Cleanup on shutdown"""

        pass
        # TODO: Add any necessary cleanup logic here"