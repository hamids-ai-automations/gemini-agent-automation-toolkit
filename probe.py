import inspect
from gemini_agent_toolkit.agent import Agent
from gemini_agent_toolkit.context import ConversationContext
print("run:", inspect.signature(Agent.run))
print("resume:", inspect.signature(Agent.resume))
print("ctx init:", inspect.signature(ConversationContext.__init__))
print("ctx methods:", [m for m in dir(ConversationContext) if not m.startswith("_")])
