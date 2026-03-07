# imports

from enum import Enum
import os
import glob
import hashlib
import json
import shutil
from contextvars import ContextVar
from threading import Lock
from typing import Dict, List, Callable, Optional
from dotenv import load_dotenv
import gradio as gr
import ollama
import torch
from transformers import AutoModelForCausalLM, GemmaTokenizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

OPENAI_MODEL = 'gpt-4o-mini'
OLLAMA_MODEL = 'llama3.2:latest'
HUGGINGFACE_MODEL = "Qwen/Qwen3-0.6B"
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = "/mnt/c/Users/crome/My Drive/documentos/obsidian/CRV/Diario"
VECTORSTORE_DIR = "vector_db"
VECTORSTORE_STATE_FILE = ".kb_state.json"


class Provider(str, Enum):
  """Proveedores de modelos disponibles en la interfaz."""

  OPENAI = "OpenAI"
  OLLAMA = "Ollama"
  HUGGINGFACE = "HuggingFace"


class PromptCaptureCallbackHandler(StdOutCallbackHandler):
  """Callback que guarda el último prompt completo enviado al modelo."""

  def __init__(self):
    super().__init__()
    self.latest_prompt = ""

  def capture_prompt(self, prompt_text: str) -> None:
    """Registra el prompt completo y lo imprime en stdout."""
    self.latest_prompt = prompt_text
    self.on_text(f"\n[Full Prompt]\n{prompt_text}\n")



def _chat_with_ollama(messages) -> Optional[str]:
  """Envía una conversación al modelo de Ollama y devuelve el texto generado."""
  response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
  return response["message"]["content"]


def _build_huggingface_prompt(messages: List[Dict]):
  """Construye un prompt simple para ejecutar el modelo causal de HuggingFace."""
  system_content = ""
  user_content = ""
  for message in messages:
    role = message.get("role")
    content = message.get("content", "")
    if role == "system":
      system_content = str(content).strip()
    if role == "user":
      user_content = str(content).strip()

  return f"""System content:
    {system_content}

    User content:
    {user_content}

    """

def _chat_with_huggingface(messages) -> Optional[str]:
  """Genera respuesta con el modelo local de HuggingFace a partir de mensajes chat."""
  # HuggingFace no consume mensajes estilo chat de forma nativa: se compactan en un prompt.
  prompt = _build_huggingface_prompt(messages)
  # Inferencia directa en el dispositivo del modelo para minimizar sobrecostes.
  inputs = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
  prompt_len = inputs["input_ids"].shape[-1]
  with torch.no_grad():
    outputs = hf_model.generate(**inputs, max_new_tokens=300)
  # Se devuelve solo la parte generada (sin repetir el prompt original).
  return hf_tokenizer.decode(outputs[0][prompt_len:],
                             skip_special_tokens=True).strip()


def _get_raw_chat_with_provider(provider: Provider) -> Callable:
  """Devuelve la función de chat raw para providers no nativos de LangChain."""
  if provider == Provider.HUGGINGFACE:
    return _chat_with_huggingface
  if provider == Provider.OLLAMA:
    return _chat_with_ollama
  raise ValueError(f"Unsupported provider: {provider}")


def _normalize_content(content: List):
  """Normaliza contenido multimodal de LangChain en una sola cadena de texto."""
  parts = []
  for item in content:
    text = item.get("text")
    parts.append(text)
  return "\n".join(part for part in parts if part).strip()


def _format_docs(docs):
  """Concatena los documentos recuperados en un bloque de contexto para el prompt."""
  return "\n\n".join(doc.page_content for doc in docs)


def _to_provider_messages(messages: List) -> List[Dict[str, str]]:
  """Convierte mensajes de LangChain al formato chat esperado por SDKs externos."""
  role_map = {"human": "user", "ai": "assistant", "system": "system"}
  provider_messages = []
  for message in messages:
    content = message.content
    if isinstance(content, list):
      content = _normalize_content(content)
    provider_messages.append({
        "role": role_map.get(message.type, "user"),
        "content": str(content).strip(),
    })
  return provider_messages


def _format_prompt_messages_for_debug(messages: List[Dict[str, str]]) -> str:
  """Formatea mensajes para visualizarlos como prompt completo en la UI."""
  lines = []
  for message in messages:
    role = message.get("role", "user").upper()
    content = message.get("content", "")
    lines.append(f"{role}:\n{content}")
  return "\n\n".join(lines).strip()


def _build_provider_llm(provider: Provider):
  """Envuelve un proveedor raw en un Runnable compatible con el pipeline LangChain."""
  chat_with_provider: Callable = _get_raw_chat_with_provider(provider)

  def _invoke(prompt_value):
    # El prompt de LangChain se transforma a lista de mensajes antes de invocar el SDK.
    messages = _to_provider_messages(prompt_value.to_messages())
    return chat_with_provider(messages)

  return RunnableLambda(_invoke)


def _get_chat_with_provider(provider: Provider):
  """Selecciona el LLM del provider en formato utilizable por LangChain."""
  if provider == Provider.OPENAI:
    # OpenAI usa integración nativa de LangChain (chat model).
    return ChatOpenAI(temperature=0.7, model=OPENAI_MODEL)
  # Ollama/HuggingFace pasan por un adapter que expone la misma interfaz de Runnable.
  return _build_provider_llm(provider)


def _resolve_project_path(path: str) -> str:
  """Resuelve una ruta relativa al directorio del script."""
  if os.path.isabs(path):
    return path
  return os.path.join(BASE_DIR, path)


def _iter_knowledge_base_files(base_path: str) -> List[str]:
  """Devuelve todos los markdown de la KB en orden estable."""
  resolved_base_path = _resolve_project_path(base_path)
  if os.path.isfile(resolved_base_path):
    return [resolved_base_path] if resolved_base_path.endswith(".md") else []
  return sorted(glob.glob(os.path.join(resolved_base_path, "**", "*.md"), recursive=True))


def _compute_knowledge_base_fingerprint(base_path: str) -> str:
  """Calcula un fingerprint de la KB usando rutas y contenido real de archivos."""
  resolved_base_path = _resolve_project_path(base_path)
  files = _iter_knowledge_base_files(base_path)
  hasher = hashlib.sha256()
  hasher.update(str(len(files)).encode("utf-8"))
  for file_path in files:
    rel_path = os.path.relpath(file_path, resolved_base_path)
    hasher.update(rel_path.encode("utf-8"))
    # Hash por contenido para evitar reindexados por cambios de mtime sin cambios reales.
    with open(file_path, "rb") as kb_file:
      hasher.update(hashlib.sha256(kb_file.read()).digest())
  return hasher.hexdigest()


def _get_vectorstore_state_path(db_name: str = VECTORSTORE_DIR) -> str:
  """Devuelve la ruta del archivo de estado del vectorstore."""
  resolved_db_name = _resolve_project_path(db_name)
  return os.path.join(resolved_db_name, VECTORSTORE_STATE_FILE)


def _load_vectorstore_state(db_name: str = VECTORSTORE_DIR) -> Dict[str, str]:
  """Carga el estado persistido del vectorstore si existe."""
  state_path = _get_vectorstore_state_path(db_name)
  if not os.path.exists(state_path):
    return {}
  with open(state_path, "r", encoding="utf-8") as state_file:
    return json.load(state_file)


def _save_vectorstore_state(kb_fingerprint: str, db_name: str = VECTORSTORE_DIR) -> None:
  """Persiste el fingerprint actual para decidir futuras reindexaciones."""
  resolved_db_name = _resolve_project_path(db_name)
  os.makedirs(resolved_db_name, exist_ok=True)
  state_path = _get_vectorstore_state_path(db_name)
  with open(state_path, "w", encoding="utf-8") as state_file:
    json.dump({"kb_fingerprint": kb_fingerprint}, state_file)


# Almacén de memoria por sesión. Cada session_id mantiene su propio historial.
chat_memory_store: Dict[str, BaseChatMessageHistory] = {}
current_prompt_handler: ContextVar[Optional[PromptCaptureCallbackHandler]] = ContextVar(
    "current_prompt_handler", default=None
)


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
  """Obtiene o crea el historial de chat asociado a una sesión."""
  if session_id not in chat_memory_store:
    chat_memory_store[session_id] = InMemoryChatMessageHistory()
  return chat_memory_store[session_id]


def _capture_full_prompt_for_debug(prompt_value):
  """Captura el prompt final antes de invocar el LLM y lo deja disponible en UI."""
  handler = current_prompt_handler.get()
  if handler is None:
    return prompt_value
  prompt_messages = _to_provider_messages(prompt_value.to_messages())
  handler.capture_prompt(_format_prompt_messages_for_debug(prompt_messages))
  return prompt_value


def _build_conversation_chain(llm, retriever):
  """Construye una cadena RAG conversacional con memoria por sesión."""
  # Prompt RAG conversacional moderno:
  # - `history` lo inyecta automáticamente RunnableWithMessageHistory
  # - `context` viene del retriever en cada turno
  prompt = ChatPromptTemplate.from_messages([
      (
          "system",
          "Usa el contexto recuperado para responder de forma precisa. "
          "Si no sabes la respuesta, dilo claramente.\n\nContexto:\n{context}",
      ),
      MessagesPlaceholder(variable_name="history"),
      ("human", "{input}"),
  ])

  retrieval_chain = (
      RunnablePassthrough.assign(
          # Recupera documentos con la pregunta actual y los serializa a contexto.
          context=RunnableLambda(lambda x: retriever.invoke(x["input"]))
          | RunnableLambda(_format_docs)
      )
      | prompt
      | RunnableLambda(_capture_full_prompt_for_debug)
      | llm
      | StrOutputParser()
  )

  # Esta capa añade memoria conversacional sin usar langchain_classic.
  return RunnableWithMessageHistory(
      retrieval_chain,
      get_session_history=_get_session_history,
      input_messages_key="input",
      history_messages_key="history",
  )


def _load_knowledge_base_documents(base_path: str) -> List[Document]:
  """Carga documentos markdown de la base de conocimiento y etiqueta su tipo."""
  resolved_base_path = _resolve_project_path(base_path)
  if not os.path.exists(resolved_base_path):
    raise ValueError(f"Knowledge base path does not exist: {resolved_base_path}")

  text_loader_kwargs = {'encoding': 'utf-8'}
  documents = []
  if os.path.isfile(resolved_base_path):
    if not resolved_base_path.endswith(".md"):
      raise ValueError(f"Knowledge base file must be .md: {resolved_base_path}")
    loader = TextLoader(resolved_base_path, encoding='utf-8')
    file_doc = loader.load()[0]
    file_doc.metadata["doc_type"] = "root"
    print(f"[KB] Indexed 1 markdown file from: {resolved_base_path}")
    print(f"[KB] - {resolved_base_path}")
    return [file_doc]

  # Carga recursiva de markdowns: funciona con archivos en raíz y en subcarpetas.
  loader = DirectoryLoader(
      resolved_base_path,
      glob="**/*.md",
      loader_cls=TextLoader,
      loader_kwargs=text_loader_kwargs
  )
  for doc in loader.load():
    source_path = doc.metadata.get("source", "")
    relative_source = os.path.relpath(source_path, resolved_base_path)
    top_level = relative_source.split(os.sep, 1)[0]
    doc.metadata["doc_type"] = "root" if os.sep not in relative_source else top_level
    documents.append(doc)

  print(f"[KB] Indexed {len(documents)} markdown files from: {resolved_base_path}")
  for doc in documents:
    source_path = doc.metadata.get("source", "")
    print(f"[KB] - {source_path}")

  return documents


def _build_vectorstore(documents, db_name: str = VECTORSTORE_DIR):
  """Construye y persiste un índice vectorial Chroma a partir de documentos."""
  if not documents:
    raise ValueError(
        "No knowledge-base documents found. Verify the path and markdown files."
    )

  resolved_db_name = _resolve_project_path(db_name)
  embeddings = OpenAIEmbeddings()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
  )
  chunks = text_splitter.split_documents(documents)
  if not chunks:
    raise ValueError("Document splitting produced no chunks to embed.")
  # Embebe y persiste los chunks en Chroma.
  return Chroma.from_documents(
      documents=chunks, embedding=embeddings, persist_directory=resolved_db_name
  )


def initialize_vectorstore(
    base_path: str = KNOWLEDGE_BASE_PATH,
    db_name: str = VECTORSTORE_DIR,
    force_rebuild: bool = False,
):
  """Inicializa vectorstore y reindexa solo si cambian los documentos."""
  current_fingerprint = _compute_knowledge_base_fingerprint(base_path)
  active_db_name = os.path.join(db_name, current_fingerprint)
  resolved_db_name = _resolve_project_path(active_db_name)
  has_index_files = os.path.exists(resolved_db_name) and bool(os.listdir(resolved_db_name))

  if (not force_rebuild) and has_index_files:
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=resolved_db_name,
        embedding_function=embeddings,
    ), current_fingerprint

  documents = _load_knowledge_base_documents(base_path=base_path)
  vectorstore = _build_vectorstore(documents, db_name=active_db_name)
  return vectorstore, current_fingerprint


def initialize_conversation_chain(provider: Provider, retriever):
  """Crea la cadena conversacional RAG para un provider específico."""
  llm = _get_chat_with_provider(provider)
  return _build_conversation_chain(llm, retriever)


# Se construye una sola vez el índice vectorial al arrancar la app.
vectorstore, kb_fingerprint = initialize_vectorstore()
# Retriever compartido por todas las cadenas de provider.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# Una cadena por provider para reutilizar configuración y memoria conversacional.
conversation_chains = {
    provider.value: initialize_conversation_chain(provider, retriever)
    for provider in Provider
}
vectorstore_refresh_lock = Lock()


def _refresh_knowledge_base_if_needed() -> None:
  """Reindexa KB y recrea cadenas de conversación cuando detecta cambios."""
  global vectorstore, retriever, conversation_chains, kb_fingerprint

  latest_fingerprint = _compute_knowledge_base_fingerprint(KNOWLEDGE_BASE_PATH)
  if latest_fingerprint == kb_fingerprint:
    return

  with vectorstore_refresh_lock:
    latest_fingerprint = _compute_knowledge_base_fingerprint(KNOWLEDGE_BASE_PATH)
    if latest_fingerprint == kb_fingerprint:
      return

    vectorstore, kb_fingerprint = initialize_vectorstore()
    retriever = vectorstore.as_retriever()
    conversation_chains = {
        provider.value: initialize_conversation_chain(provider, retriever)
        for provider in Provider
    }


def reset_vectorstore_and_chat_memory():
  """Borra Chroma, reindexa KB desde cero y limpia la memoria de chat en servidor."""
  global vectorstore, retriever, conversation_chains, kb_fingerprint

  with vectorstore_refresh_lock:
    try:
      active_db_dir = _resolve_project_path(os.path.join(VECTORSTORE_DIR, kb_fingerprint))
      if os.path.exists(active_db_dir):
        try:
          shutil.rmtree(active_db_dir)
        except Exception as delete_exc:
          print(f"[RESET][WARN] Could not delete {active_db_dir}: {delete_exc}")

      # Tras el borrado manual (si fue posible), forzamos rebuild de la KB actual.
      vectorstore, kb_fingerprint = initialize_vectorstore(force_rebuild=True)
      retriever = vectorstore.as_retriever()
      conversation_chains = {
          provider.value: initialize_conversation_chain(provider, retriever)
          for provider in Provider
      }
      chat_memory_store.clear()
      return "", "Base vectorial recreada y memoria de chat reiniciada."
    except Exception as exc:
      print(f"[RESET][ERROR] {exc}")
      return "", f"Error en reset: {exc}"


def chat(message: str, history: List[Dict], provider: str):
  """Maneja cada turno de chat en Gradio usando el provider seleccionado."""
  # Antes de responder, comprobamos si cambió la KB para refrescar Chroma automáticamente.
  _refresh_knowledge_base_if_needed()
  _ = history
  selected_provider = Provider(provider)
  conversation_chain = conversation_chains[selected_provider.value]
  prompt_handler = PromptCaptureCallbackHandler()
  context_token = current_prompt_handler.set(prompt_handler)

  try:
    # Session id separado por provider para no mezclar historiales entre modelos.
    response = conversation_chain.invoke(
        {"input": message},
        config={
            "configurable": {"session_id": f"default-{selected_provider.value}"},
            "callbacks": [prompt_handler],
        },
    )
  finally:
    current_prompt_handler.reset(context_token)

  return response, prompt_handler.latest_prompt


# Salida auxiliar para inspeccionar el prompt completo enviado al modelo.
prompt_debug_output = gr.Textbox(
    label="Prompt completo enviado al modelo",
    lines=28,
    max_lines=40,
    interactive=False,
    autoscroll=False,
    render=False,
)
reset_status_output = gr.Textbox(
    label="Estado reset",
    lines=2,
    interactive=False,
    render=False,
)

# Interfaz en dos columnas: chat a la izquierda, prompt capturado a la derecha.
with gr.Blocks() as interface:
  with gr.Row():
    with gr.Column(scale=3):
      gr.ChatInterface(
          fn=chat,
          additional_inputs=[
              gr.Dropdown(
                  choices=[provider.value for provider in Provider],
                  value=Provider.OLLAMA.value,
                  label="Provider",
              )
          ],
          additional_outputs=[prompt_debug_output],
      )
    with gr.Column(scale=2):
      reset_button = gr.Button("Reset KB + memoria chat", variant="stop")
      reset_button.click(
          fn=reset_vectorstore_and_chat_memory,
          outputs=[prompt_debug_output, reset_status_output],
      )
      prompt_debug_output.render()
      reset_status_output.render()

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
# - en el terminal ejecuto "gradio week5/week5_exercise.py"
# - los cambios en el html se actualizan automáticamente
if __name__ == "__main__":
  interface.launch(server_name="0.0.0.0", server_port=7863, inbrowser=True)
