from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer
from semantic_router.index.qdrant import QdrantIndex
from qdrant_client import QdrantClient
from app.core.config import settings

class RouterService:
    def __init__(self):
        # Initialize Encoder (using a standard one for now, or could be OpenAI/Cohere)
        # Using HuggingFace for cost efficiency in this example, but can be swapped.
        self.encoder = HuggingFaceEncoder()
        
        # Connect to Qdrant
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # Initialize Index
        self.index = QdrantIndex(
            client=self.qdrant_client,
            collection_name=settings.QDRANT_COLLECTION,
        )
        
        # Initialize basic routes just to have structure, 
        # but the real power comes from the seeded index.
        # If the index is already populated, we might not need to pass routes list here 
        # depending on semantic-router version, but usually we define the Route objects.
        # For dynamic checking against what's in Qdrant:
        self.routes = [] 
        
        # We assume the index is already populated via seed_router.py
        # The RouteLayer needs at least definitions of the routes if we want to use them locally 
        # or we trust the Index to return the route name.
        
        self.layer = RouteLayer(encoder=self.encoder, index=self.index, routes=self.routes)

    async def check_route(self, text: str) -> str | None:
        """
        Checks the semantic route for the given text.
        Returns the route name if found, else None.
        """
        try:
            # semantic-router call
            route = self.layer(text)
            if route.name:
                return route.name
            return None
        except Exception as e:
            print(f"Router Error: {e}")
            return None
