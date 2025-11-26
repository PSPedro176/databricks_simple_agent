# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definição centralizada de catálogos, schemas, prompts, etc

# COMMAND ----------

# DBTITLE 1,Bibliotecas
import mlflow
import pandas

# COMMAND ----------

# DBTITLE 1,Variáveis
# TODO: Atualizar
CATALOG = 'perdomo_demos'
SCHEMA = 'demo_agents'
#################

PROMPT_NAME = f'{CATALOG}.{SCHEMA}.demo_prompt'
EVAL_DATASET_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data/eval_dataset"
EVAL_DATASET_TABLE = f"{CATALOG}.{SCHEMA}.eval_dataset"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.simple_model"

UC_TOOL_NAMES = [
    f"{CATALOG}.{SCHEMA}.get_customer_by_email",
    f"{CATALOG}.{SCHEMA}.get_billing_and_subs"
]

INPUT_EXAMPLE = {"input": [{
            "role": "user", "content": "What is the full name and address of the customer with the email john21@example.net?"
}]}


# COMMAND ----------

# DBTITLE 1,Registro de prompt
text = r"Seu trabalho é fornecer ajuda ao cliente. Chame as ferramentas para responder. Responda no mesmo idioma da pergunta."

prompt_v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=text,
    commit_message="Prompt simplificado"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Código e teste do modelo
# MAGIC ###### Lembrete: alterar catálogo e schema no código abaixo do modelo

# COMMAND ----------

# DBTITLE 1,Definição do agente
# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC from typing import Any, Callable, Generator, Optional
# MAGIC from uuid import uuid4
# MAGIC import warnings
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Atualizar
# MAGIC CATALOG = 'perdomo_demos'
# MAGIC SCHEMA = 'demo_agents'
# MAGIC #################
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC PROMPT_NAME = f'{CATALOG}.{SCHEMA}.demo_prompt'
# MAGIC UC_TOOL_NAMES = [
# MAGIC     f"{CATALOG}.{SCHEMA}.get_customer_by_email",
# MAGIC     f"{CATALOG}.{SCHEMA}.get_billing_and_subs"
# MAGIC ]
# MAGIC SYSTEM_PROMPT = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{PROMPT_NAME}/3").template
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     Class representing a tool for the agent.
# MAGIC     - "name" (str): The name of the tool.
# MAGIC     - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
# MAGIC     - "exec_fn" (Callable): Function that implements the tool logic
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC
# MAGIC def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC
# MAGIC     # Define a wrapper that accepts kwargs for the UC tool call,
# MAGIC     # then passes them to the UC tool execution client
# MAGIC     def exec_fn(**kwargs):
# MAGIC         function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         if function_result.error is not None:
# MAGIC             return function_result.error
# MAGIC         else:
# MAGIC             return function_result.value
# MAGIC     return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)
# MAGIC
# MAGIC
# MAGIC TOOL_INFOS = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC # TODO: Add additional tools
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(create_tool_info(tool_spec))
# MAGIC
# MAGIC class ToolCallingAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Class representing a tool-calling Agent
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """Initializes the ToolCallingAgent with tools."""
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = (
# MAGIC             self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         )
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """Returns tool specifications in the format OpenAI expects."""
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """Executes the specified tool with the given arguments."""
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.model_serving_client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 tools=self.get_tool_specs(),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 chunk_dict = chunk.to_dict()
# MAGIC                 if len(chunk_dict.get("choices", [])) > 0:
# MAGIC                     yield chunk_dict
# MAGIC
# MAGIC     def handle_tool_call(
# MAGIC         self,
# MAGIC         tool_call: dict[str, Any],
# MAGIC         messages: list[dict[str, Any]],
# MAGIC     ) -> ResponsesAgentStreamEvent:
# MAGIC         """
# MAGIC         Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
# MAGIC         """
# MAGIC         args = json.loads(tool_call["arguments"])
# MAGIC         result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
# MAGIC
# MAGIC         tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
# MAGIC         messages.append(tool_call_output)
# MAGIC         return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         messages: list[dict[str, Any]],
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         for _ in range(max_iter):
# MAGIC             last_msg = messages[-1]
# MAGIC             if last_msg.get("role", None) == "assistant":
# MAGIC                 return
# MAGIC             elif last_msg.get("type", None) == "function_call":
# MAGIC                 yield self.handle_tool_call(last_msg, messages)
# MAGIC             else:
# MAGIC                 yield from output_to_responses_items_stream(
# MAGIC                     chunks=self.call_llm(messages), aggregator=messages
# MAGIC                 )
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
# MAGIC         )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         if SYSTEM_PROMPT:
# MAGIC             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
# MAGIC         yield from self.call_and_run_tools(messages=messages)
# MAGIC
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Testando agente
from agent import AGENT
AGENT.predict(INPUT_EXAMPLE)

# COMMAND ----------

# DBTITLE 1,Inicializando experimento/criando nova run
# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=INPUT_EXAMPLE,
        pip_requirements=[
            "databricks-openai",
            "backoff",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Avaliação

# COMMAND ----------

# DBTITLE 1,Recuperando dados simulados de avaliação
eval_example = spark.read.json(EVAL_DATASET_VOLUME)
display(eval_example)

# COMMAND ----------

# DBTITLE 1,Criando um dataset de avaliação
try:
  eval_dataset = mlflow.genai.datasets.get_dataset(EVAL_DATASET_TABLE)
except Exception as e:
  eval_dataset = mlflow.genai.datasets.create_dataset(EVAL_DATASET_TABLE)
  eval_dataset.merge_records(eval_example)
  print("Added records to the evaluation dataset.")

# Preview the dataset
display(eval_dataset.to_df())

# COMMAND ----------

# DBTITLE 1,Definindo scorers
from mlflow.genai import scorers
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, scorer
from mlflow.entities import Trace, Feedback, SpanType

# Opção 1: Avaliadores padrão pré-construídos
standard_scores = [Safety(), RelevanceToQuery()] # Outros: RetrievalRelevance, RetrievalGroundedness

# Opção 2: Avaliadores para diretrizes personalizadas
personalized_score_1 = Guidelines(
    guidelines="The response must be in the same language of the question.",
    name="same_language",
)
personalized_score_2 = Guidelines(
    guidelines="The question must contain an e-mail using an '@'.",
    name="has_email",
)

personalized_scores = [personalized_score_1, personalized_score_2]

# Opção 3: Buscar avaliadores definidos na interface do usuário
has_data = scorers.list_scorers()[0]

# Opção 4: Avaliadores definidos usando funções
@scorer
def acceptable_response_time(trace: Trace) -> Feedback:
    llm_span = trace.search_spans(span_type=SpanType.CHAT_MODEL)[0]
    response_time = (llm_span.end_time_ns - llm_span.start_time_ns) / 1e9
    max_duration = 5.0

    if response_time <= max_duration:
        v = "yes"
        r = f"LLM response time {response_time:.2f}s is within the {max_duration}s limit."
    else:
        v = "no"
        r = f"LLM response time {response_time:.2f}s exceeds the {max_duration}s limit."

    return Feedback(value=v, rationale=r)
    
# Lista de scorers
agent_scorers = standard_scores + personalized_scores + [acceptable_response_time] + [has_data]

# COMMAND ----------

# DBTITLE 1,Função de avaliação
loaded_model = mlflow.pyfunc.load_model(f"runs:/{logged_agent_info.run_id}/agent")

def predict_wrapper(question):
    model_input = pandas.DataFrame({"input": [[{"role": "user", "content": question}]]})
    response = loaded_model.predict(model_input)
    return response['output'][-1]['content'][-1]['text']

# COMMAND ----------

# DBTITLE 1,Avaliação
results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_wrapper, scorers=agent_scorers)

# COMMAND ----------

# DBTITLE 1,Criando labeling session
session = mlflow.genai.labeling.create_labeling_session(
    name="model_session_feedback",
    assigned_users=["pedro.perdomo@databricks.com"],
    label_schemas=["feedback", "nps"]
)

traces_df = mlflow.search_traces(run_id=results.run_id)
session.add_traces(traces_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registrando modelo no Unity Catalog

# COMMAND ----------

# DBTITLE 1,Validação
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Registro
mlflow.set_registry_uri("databricks-uc")

uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# DBTITLE 1,Deploy
from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})
