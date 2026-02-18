# imports

from enum import Enum
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import ollama

OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3.2:latest"

class Modelo(str, Enum):
  OPENAI = "OpenAI"
  OLLAMA = "Ollama"


def get_system_message():
  return '''
    Eres un asistente especializado en generar datos sintéticos para diferentes campos. Cuando el usuario te proporcione un campo y una descripción, debes generar 5 ejemplos de datos sintéticos que se ajusten a esa descripción. Asegúrate de que los datos sean variados y realistas, siguiendo la descripción proporcionada por el usuario.
    IMPORTANTE: responde en formato JSON. Por ejemplo:

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


def _get_chat_with_provider(provider: Modelo):
  if provider == Modelo.OPENAI:
    return _chat_with_openai
  return _chat_with_ollama


def _normalize_content(content):
  if isinstance(content, str):
    return content

  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str):
          parts.append(text)
    return "\n".join(part for part in parts if part).strip()

  if isinstance(content, dict):
    text = content.get("text")
    if isinstance(text, str):
      return text

  if content is None:
    return ""

  return str(content)


def _build_messages(message, history):
  messages = [{"role": "system", "content": get_system_message()}]

  for item in (history or []):
    if isinstance(item, dict):
      role = item.get("role")
      content = _normalize_content(item.get("content"))
      if role in {"user", "assistant"} and content:
        messages.append({"role": role, "content": content})
      continue

    if isinstance(item, (list, tuple)) and len(item) == 2:
      user_msg, assistant_msg = item
      user_text = _normalize_content(user_msg)
      assistant_text = _normalize_content(assistant_msg)
      if user_text:
        messages.append({"role": "user", "content": user_text})
      if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})

  user_content = _normalize_content(message)
  messages.append({"role": "user", "content": user_content})
  return messages


def chat(message, history, provider):
  selected_provider = provider if isinstance(provider, Modelo) else Modelo(provider)
  messages = _build_messages(message, history)
  response = _get_chat_with_provider(selected_provider)(messages)
  return response


provider_dropdown = gr.Dropdown(
    choices=[m.value for m in Modelo],
    value=Modelo.OLLAMA.value,
    label="Proveedor de modelo",
)
interfaz = gr.ChatInterface(
    # Funcion principal:
    fn=chat,
    # Nuevos inputs para la función de chat:
    additional_inputs=[provider_dropdown],
    # Personalizamos el selector:
    additional_inputs_accordion=gr.Accordion(
        "Configuración de modelo", open=True
    ),
)

# - declaro una variable global "interfaz"
# - en el main cargo el .env y lanzo la interfaz
# - lanzo la interfaz en el puerto 7863 que tengo habilitado el microfono
# - en el terminal ejecuto "gradio week3/week3_exercise.py"
# - los cambios en el html se actualizan automáticamente
if __name__ == "__main__":
  load_dotenv()
  interfaz.launch(server_name="0.0.0.0", server_port=7863, inbrowser=True)
