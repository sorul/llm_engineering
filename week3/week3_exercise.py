# imports

from enum import Enum
from typing import Dict, List
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import ollama
from transformers import pipeline

OPENAI_MODEL = 'gpt-4o-mini'
OLLAMA_MODEL = 'llama3.2:latest'
HUGGINGFACE_MODEL = 'Qwen/Qwen3-0.6B'


class Proveedor(str, Enum):
  OPENAI = "OpenAI"
  OLLAMA = "Ollama"
  HUGGINGFACE = "HuggingFace"


def get_system_message():
  return '''
    Eres un asistente especializado en generar datos sintéticos para diferentes campos. Cuando el usuario te proporcione un campo y una descripción, debes generar 5 ejemplos de datos sintéticos que se ajusten a esa descripción. Asegúrate de que los datos sean variados y realistas, siguiendo la descripción proporcionada por el usuario.
    IMPORTANTE: responde directamente en formato JSON. Por ejemplo:

    User: Dame productos de Amazon.
    Assistant: [
        {"nombre": "Echo Dot", "categoría": "Electrónica", "precio": 49.99},
        {"nombre": "Kindle Paperwhite", "categoría": "Electrónica", "precio": 129.99},
        {"nombre": "Fire TV Stick", "categoría": "Electrónica", "precio": 39.99},
    ]
  '''


def _chat_with_openai(messages):
  openai_client = OpenAI()
  response = openai_client.chat.completions.create(
      model=OPENAI_MODEL, messages=messages
  )
  return response.choices[0].message.content


def _chat_with_ollama(messages):
  response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
  return response["message"]["content"]


def _chat_with_huggingface(messages):
  response = hf_model(messages)
  generated = response[0].get("generated_text")
  last_message = generated[-1]
  content = last_message.get("content")
  return content


def _get_chat_with_provider(provider: Proveedor):
  if provider == Proveedor.OPENAI:
    return _chat_with_openai
  if provider == Proveedor.HUGGINGFACE:
    return _chat_with_huggingface
  return _chat_with_ollama


def _normalize_content(content: List):
  parts = []
  for item in content:
    text = item.get("text")
    parts.append(text)
  return "\n".join(part for part in parts if part).strip()


def _build_messages(message: str, history: List[Dict] = []):
  messages = [{"role": "system", "content": get_system_message()}]

  for item in history:
    role = item.get("role")
    content = _normalize_content(item.get("content", []))
    if role in {"user", "assistant"} and content:
      messages.append({"role": role, "content": content})

  messages.append({"role": "user", "content": message})
  return messages


def chat(message: str, history: List[Dict], provider: str):
  selected_provider = Proveedor(provider)
  messages = _build_messages(message, history)
  response = _get_chat_with_provider(selected_provider)(messages)
  return response


provider_dropdown = gr.Dropdown(
    choices=[m.value for m in Proveedor],
    value=Proveedor.OLLAMA.value,
    label="Proveedor de modelo",
)
interface = gr.ChatInterface(
    # Funcion principal:
    fn=chat,
    # Nuevos inputs para la función de chat:
    additional_inputs=[provider_dropdown],
    # Personalizamos el selector:
    additional_inputs_accordion=gr.Accordion(
        "Configuración de modelo", open=True
    ),
)
# Necesario para cachear el pipeline de HuggingFace y no cargarlo en cada llamada:
hf_model = pipeline("text-generation", model=HUGGINGFACE_MODEL)

# - declaro una variable global "interfaz"
# - en el main cargo el .env y lanzo la interfaz
# - lanzo la interfaz en el puerto 7863
# - en el terminal ejecuto "gradio week3/week3_exercise.py"
# - los cambios en el html se actualizan automáticamente
if __name__ == "__main__":
  load_dotenv()
  interface.launch(server_name="0.0.0.0", server_port=7863, inbrowser=True)
