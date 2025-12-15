from langfuse import Langfuse
from app.core.config import settings
from typing import List, Dict, Any, Optional

# Initializing Langfuse client
langfuse = Langfuse(
    secret_key=settings.LANGFUSE_SECRET_KEY,
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    host=settings.LANGFUSE_HOST
)

class BaseNode:
    def __init__(self, name: str, functions: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.functions = functions or []

    def get_system_message(self):
        """
        Fetches the system prompt from Langfuse. 
        """
        prompt_name = f"{self.name.lower()}_system"
        try:
            # In a real app, ensure this doesn't block if Langfuse is slow
            # prompt = langfuse.get_prompt(prompt_name)
            # return prompt.compile()
            
            # Fallback/Simulation for Development without active Langfuse prompts
            return self._get_default_prompt()
            
        except Exception as e:
            print(f"Langfuse Warning: Could not fetch prompt '{prompt_name}'. Using default. Error: {e}")
            return self._get_default_prompt()

    def _get_default_prompt(self):
        return f"You are a useful assistant in the {self.name} state."

class RootGreetingNode(BaseNode):
    def __init__(self):
        super().__init__("root_greeting")
    
    def _get_default_prompt(self):
        return (
            "Eres Giuliana, la recepcionista del estudio del Dr. Sanchez. "
            "Saluda cordialmente, menciona al estudio y pregunta en qué puedes ayudar. "
            "Tu objetivo es escuchar al usuario para entender su problema legal. "
            "Sé breve y profesional."
        )

class QualifyStartNode(BaseNode):
    def __init__(self):
        super().__init__("qualify_start")

    def _get_default_prompt(self):
        return (
            "El usuario tiene un problema de tránsito. Debes filtrar la localidad. "
            "Pregunta: '¿En qué localidad ocurrió el accidente?' y '¿Cuál es su nombre?'. "
            "Si es en Córdoba Capital o alrededores, pasaremos al siguiente paso. "
            "Si es muy lejos, rechazaremos el caso amablemente. "
            "Obtén estos datos."
        )

class QualifyDetailsNode(BaseNode):
    def __init__(self):
        super().__init__("qualify_details")

    def _get_default_prompt(self):
        return (
            "El usuario está en una localidad válida. "
            "Ahora obtén detalles críticos: "
            "1. ¿Hubo heridos graves o fallecidos? "
            "2. Una breve descripción del hecho. "
            "No des consejos legales, solo recaba información."
        )

class OfferAppointmentNode(BaseNode):
    def __init__(self):
        funcs = [{
            "name": "check_availability",
            "description": "Checks if the doctor is available for a call or appointment.",
            "parameters": {"type": "object", "properties": {}}
        }]
        super().__init__("offer_appointment", functions=funcs)

    def _get_default_prompt(self):
        return (
            "El Dr. Sanchez puede tomar el caso. "
            "El estudio está en Sarachaga 1234. La consulta inicial son $20.000 pesos en efectivo. "
            "¿Desea que busquemos un horario para que venga?"
        )

class BookingProcessNode(BaseNode):
    def __init__(self):
        funcs = [{
            "name": "book_appointment",
            "description": "Books the appointment.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "date": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["date", "email"]
            }
        }]
        super().__init__("booking_process", functions=funcs)

    def _get_default_prompt(self):
        return (
            "Coordina fecha y hora exacta. Pide un email para confirmación. "
            "Usa la herramienta book_appointment cuando tengas los datos."
        )

class RejectionScopeNode(BaseNode):
    def __init__(self):
        super().__init__("rejection_scope")

    def _get_default_prompt(self):
        return (
            "El Dr. Sanchez se especializa exclusivamente en accidentes de tránsito. "
            "Explica esto cortésmente y pregunta si desea dejar un mensaje de todos modos."
        )

class RejectionLocationNode(BaseNode):
    def __init__(self):
        super().__init__("rejection_location")

    def _get_default_prompt(self):
        return (
            "Por el momento el Dr. solo litiga en tribunales de la ciudad de Córdoba. "
            "Explica esto cortésmente y pregunta si desea dejar un mensaje."
        )

class TransferLogicNode(BaseNode):
    def __init__(self):
        super().__init__("transfer_logic")

    def _get_default_prompt(self):
        return (
            "Simula verificar la disponibilidad. "
            "El Dr. está en audiencia. "
            "Di: 'El Dr. está en audiencia. ¿Prefiere agendar una cita o dejar un mensaje?'"
        )
