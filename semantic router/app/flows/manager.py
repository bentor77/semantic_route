from typing import Dict, List, AsyncGenerator
from app.services.llm_service import SmartLLMService
from app.services.router_service import RouterService
from app.flows.nodes import (
    RootGreetingNode, QualifyStartNode, QualifyDetailsNode,
    OfferAppointmentNode, BookingProcessNode, RejectionScopeNode,
    RejectionLocationNode, TransferLogicNode, BaseNode
)

class FlowInstance:
    def __init__(self, call_id: str, llm_service: SmartLLMService):
        self.call_id = call_id
        self.llm_service = llm_service
        self.current_node: BaseNode = RootGreetingNode()
        self.history: List[Dict] = []
        self.data: Dict = {} # Store gathered info like local identification
    
    async def process_input(self, text: str) -> str:
        """Non-streaming wrapper"""
        full_resp = ""
        async for chunk in self.process_input_stream(text):
            full_resp += chunk
        return full_resp

    async def process_input_stream(self, text: str) -> AsyncGenerator[str, None]:
        # 1. Helper to determine transition BEFORE generating response (if possible)
        # or we might want to let the LLM generate first.
        # For RootGreeting, we rely on Router.
        
        transitioned_node = await self.check_router_transition(text)
        if transitioned_node:
            self.current_node = transitioned_node
            # We might want to clear history or keep it? 
            # Usually keep it so LLM knows context, but with different system prompt.
        
        # 2. Get System Message
        system_message = self.current_node.get_system_message()
        
        # 2.1 Get Tools if any
        tools = None
        if self.current_node.functions:
            tools = []
            for func in self.current_node.functions:
                tools.append({
                    "type": "function",
                    "function": func
                })

        # 3. Stream Response
        full_response = ""
        async for chunk in self.llm_service.get_response_stream(
            text=text, 
            system_message=system_message, 
            history=self.history,
            tools=tools
        ):
            full_response += chunk
            yield chunk
        
        # 4. Update History
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": full_response})
        
        # 5. Post-Response Transition Checks (Logic based on user input content or flow state)
        self.check_post_interaction_transition(text, full_response)

    async def check_router_transition(self, text: str) -> BaseNode | None:
        """
        Uses Semantic Router to determine if we should jump from Root to somewhere else.
        """
        # Only active in Root usually, or global interrupts like "Transfer"
        # Global Interrupts:
        route = await self.llm_service.router.check_route(text)
        
        if route == "human_handoff":
            return TransferLogicNode()
        
        if isinstance(self.current_node, RootGreetingNode):
            if route == "legal_issue_traffic":
                return QualifyStartNode()
            # If route == "pricing_info": return OfferAppointmentNode() # Maybe?
            
        return None

    def check_post_interaction_transition(self, user_text: str, assistant_response: str):
        """
        Heuristic transition logic based on conversation state.
        Real implementation would use structured LLM outputs or specific classifiers.
        """
        text_lower = user_text.lower()
        
        # Node B -> Node C or F (Location check)
        if isinstance(self.current_node, QualifyStartNode):
            # Keyword heuristics for demo
            if "córdoba" in text_lower or "cordoba" in text_lower or "capital" in text_lower:
                self.current_node = QualifyDetailsNode()
            elif "buenos aires" in text_lower or "rosario" in text_lower or "lejos" in text_lower:
                self.current_node = RejectionLocationNode()
        
        # Node C -> Node D (After user gives details)
        elif isinstance(self.current_node, QualifyDetailsNode):
            # Assume if user wrote a decent length message or mentioned "heridos", "auto", etc, we proceed.
            # Simple counter: assume 1-2 turns. For now, immediate transition after 1 answer? 
            # Prompt asked for 2 questions. Let's assume we stay 1 turn then move. 
            # Hard to track turns without 'turn_count' in node. 
            # Let's just move after any input for this demo, or specifically if length > 5 chars.
            if len(user_text) > 3:
                self.current_node = OfferAppointmentNode()

        # Node D -> Node E (Booking)
        elif isinstance(self.current_node, OfferAppointmentNode):
            if any(x in text_lower for x in ["sí", "si", "quiero", "agendar", "cita"]):
                self.current_node = BookingProcessNode()

class FlowManager:
    def __init__(self):
        self.router_service = RouterService()
        self.llm_service = SmartLLMService(router_service=self.router_service)
        self.active_flows: Dict[str, FlowInstance] = {}

    def get_or_create_flow(self, call_id: str) -> FlowInstance:
        if call_id not in self.active_flows:
            self.active_flows[call_id] = FlowInstance(call_id, self.llm_service)
        return self.active_flows[call_id]

flow_manager = FlowManager()
