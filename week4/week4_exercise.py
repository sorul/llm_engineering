# imports

from enum import Enum
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
import ollama
import torch
from transformers import AutoModelForCausalLM, GemmaTokenizer

load_dotenv()

OPENAI_MODEL = 'gpt-4o-mini'
OLLAMA_MODEL = 'codellama:7b'
HUGGINGFACE_MODEL = "google/codegemma-2b"
HF_TOKEN = os.getenv("HF_TOKEN")


class Provider(str, Enum):
  OPENAI = "OpenAI"
  OLLAMA = "Ollama"
  HUGGINGFACE = "HuggingFace"


def get_system_message():
  return '''
    Eres un asistente especializado en generar test unitarios. El usuario proporcionará una función de Python y tú deberás generar 2 casos de test unitarios utilizando el framework unittest. Responde directamente con el código de los test unitarios sin explicaciones adicionales. Asegúrate de cubrir casos típicos y casos límite en tus tests. No importes librerías adicionales. Da por hecho que tienes acceso a la función que se va a testear.
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
  prompt = _build_huggingface_prompt(messages)
  inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
  prompt_len = inputs["input_ids"].shape[-1]
  with torch.no_grad():
    outputs = hf_model.generate(**inputs, max_new_tokens=300)
  return hf_tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def _get_chat_with_provider(provider: Provider):
  if provider == Provider.OPENAI:
    return _chat_with_openai
  if provider == Provider.HUGGINGFACE:
    return _chat_with_huggingface
  return _chat_with_ollama


def _normalize_content(content: List):
  parts = []
  for item in content:
    text = item.get("text")
    parts.append(text)
  return "\n".join(part for part in parts if part).strip()


def _build_huggingface_prompt(messages: List[Dict]):
  system_content = ""
  user_content = ""
  for message in messages:
    role = message.get("role")
    content = message.get("content", "")
    if role == "system":
      system_content = str(content).strip()
    if role == "user":
      user_content = str(content).strip()

  return f"""Instrucciones:
{system_content}

Codigo Python del usuario:
{user_content}

Devuelve unicamente el codigo de tests usando unittest:
"""


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
  selected_provider = Provider(provider)
  messages = _build_messages(message, history)
  response = _get_chat_with_provider(selected_provider)(messages)
  return response


def _get_python_code_level_1():
  return '''
  def is_odd(n):
    """Devuelve True si n es impar, False si es par."""
    return n % 2 == 1
  '''


def _get_python_code_level_2():
  return '''
  def normalize_email(email):
    """
    Normaliza un email:
    - quita espacios en blanco al inicio y final
    - pasa todo a minúsculas
    - valida que contenga un único "@"
    """
    if email is None:
      raise TypeError("email no puede ser None")

    cleaned = email.strip().lower()
    if cleaned.count("@") != 1:
      raise ValueError("email inválido")

    local_part, domain = cleaned.split("@")
    if not local_part or not domain or "." not in domain:
      raise ValueError("email inválido")

    return cleaned
  '''

def create_unit_test(code: str, provider: str):
  return chat(code, [], provider)


def _strip_markdown_code_fences(content: str):
  if not content:
    return content

  python_fenced_blocks = re.findall(
      r"```python\s*\n([\s\S]*?)```",
      content,
      flags=re.IGNORECASE,
  )
  if python_fenced_blocks:
    clean_content = "\n\n".join(
        block.strip() for block in python_fenced_blocks if block.strip()
    )
  else:
    fenced_blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", content)
    clean_content = "\n\n".join(
        block.strip() for block in fenced_blocks if block.strip()
    ) if fenced_blocks else content.strip()

  # Elimina tokens especiales de modelos FIM (p. ej. <|file_separator|>)
  clean_content = re.sub(r"<\|[^>\n]+\|>", "", clean_content)
  return clean_content.strip()


def execute_tests(code: str, tests: str):
  if not tests or not tests.strip():
    return "No hay tests para ejecutar."

  clean_tests = _strip_markdown_code_fences(tests)
  combined_code = f"{code.strip()}\n\n{clean_tests}\n"

  with tempfile.TemporaryDirectory() as tmp_dir:
    test_file_path = f"{tmp_dir}/test_generated.py"
    with open(test_file_path, "w", encoding="utf-8") as test_file:
      test_file.write(combined_code)

    try:
      result = subprocess.run(
          [
              sys.executable,
              "-m",
              "unittest",
              "discover",
              "-s",
              tmp_dir,
              "-p",
              "test_generated.py",
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      output = "\n".join(
          part for part in [result.stdout.strip(), result.stderr.strip()] if part
      )
      return output or "Ejecucion completada sin salida."
    except FileNotFoundError:
      return "No se encontro el ejecutable de Python en el entorno."


with gr.Blocks() as interface:
  with gr.Row():
    python_code_input = gr.Textbox(
        label="Codigo Python",
        value=_get_python_code_level_2(),
        lines=18,
    )
    unit_test_output = gr.Textbox(
        label="Resultado del bot",
        lines=18,
    )
    execution_output = gr.Textbox(
        label="Resultado de ejecutar tests",
        lines=18,
    )

  with gr.Row():
    create_test_button = gr.Button(
        "Crear test unitario",
        variant="primary",
        scale=1,
    )
    execute_tests_button = gr.Button(
        "Ejecutar tests",
        scale=1,
    )

  provider_dropdown = gr.Dropdown(
      choices=[m.value for m in Provider],
      value=Provider.OLLAMA.value,
      label="Provider de modelo",
  )

  create_test_button.click(
      fn=create_unit_test,
      inputs=[python_code_input, provider_dropdown],
      outputs=unit_test_output,
  )
  execute_tests_button.click(
      fn=execute_tests,
      inputs=[python_code_input, unit_test_output],
      outputs=execution_output,
  )

# Necesario para cachear el modelo de HuggingFace y no cargarlo en cada llamada:
hf_tokenizer = GemmaTokenizer.from_pretrained(
    HUGGINGFACE_MODEL,
    token=HF_TOKEN,
)
hf_model = AutoModelForCausalLM.from_pretrained(
    HUGGINGFACE_MODEL,
    token=HF_TOKEN,
)

# - declaro una variable global "interfaz"
# - en el main cargo el .env y lanzo la interfaz
# - lanzo la interfaz en el puerto 7863
# - en el terminal ejecuto "gradio week4/week4_exercise.py"
# - los cambios en el html se actualizan automáticamente
if __name__ == "__main__":
  interface.launch(server_name="0.0.0.0", server_port=7863, inbrowser=True)
