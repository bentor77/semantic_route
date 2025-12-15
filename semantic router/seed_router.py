import os
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer
from semantic_router.index.qdrant import QdrantIndex
from qdrant_client import QdrantClient
from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

def seed():
    print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    
    # Define Routes with utterances
    handoff_route = Route(
        name="human_handoff",
        utterances=[
            "quiero hablar con un humano",
            "pásame con una persona",
            "no eres real",
            "dame con un agente",
            "necesito soporte real",
            "transferirme"
        ]
    )
    
    traffic_route = Route(
        name="legal_issue_traffic",
        utterances=[
            "tuve un accidente de tránsito",
            "me chocaron el auto",
            "choque en la ruta",
            "necesito un abogado por un accidente",
            "tengo un problema legal de transito",
            "accidente con lesionados",
            "me atropellaron"
        ]
    )

    pricing_route = Route(
        name="pricing_info",
        utterances=[
            "cuánto cuesta",
            "cuál es el precio",
            "tienen planes gratuitos?",
            "dime las tarifas",
            "costo del servicio",
            "honorarios"
        ]
    )
    
    routes = [handoff_route, traffic_route, pricing_route]

    # Initialize Encoder (must match what is used in app)
    encoder = HuggingFaceEncoder()
    
    # Initialize Qdrant Client
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    
    # Initialize Index
    # Note: If collection exists, we might want to recreate it to be clean, 
    # but RouteLayer usually handles upserts.
    index = QdrantIndex(
        client=qdrant_client, 
        collection_name=settings.QDRANT_COLLECTION,
    )
    
    # Create RouteLayer to trigger embedding and indexing
    print("Indexing routes...")
    rl = RouteLayer(encoder=encoder, index=index, routes=routes)
    
    print("Seeding complete!")
    
    # Quick Test
    test_phrase = "quiero saber los precios"
    res = rl(test_phrase)
    print(f"Test '{test_phrase}' -> {res.name}")

if __name__ == "__main__":
    seed()
