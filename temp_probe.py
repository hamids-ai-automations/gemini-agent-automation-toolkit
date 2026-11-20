import inspect
from google import genai
from google.genai import types
print("genai", genai.__version__ if hasattr(genai,"__version__") else "?")
print("Client init:", str(inspect.signature(genai.Client.__init__))[:400])
print("models.generate_content:", str(inspect.signature(genai.models.Models.generate_content))[:400])
cfg = [f for f in types.GenerateContentConfig.model_fields]
print("Config fields:", cfg)
print("FunctionDeclaration fields:", list(types.FunctionDeclaration.model_fields))
print("FunctionCall fields:", list(types.FunctionCall.model_fields))
print("Tool fields:", list(types.Tool.model_fields))
print("Part fields:", list(types.Part.model_fields))
print("Content fields:", list(types.Content.model_fields))
print("UsageMerge:", list(types.GenerateContentResponseUsageMetadata.model_fields))
print("Response fields:", list(types.GenerateContentResponse.model_fields))
print("HttpOptions:", list(types.HttpOptions.model_fields))
print("Errors module:", [n for n in dir(types) if 'Error' in n][:10])
