from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from uuid import uuid4
import os



# setting up prompt with chat history
SYSTEM_PROMPT = """\
You are Minerva, a conversational assistant, skilled at crafting responses to \
effectively communcate with a specific user given some information about them. \
Do NOT generate human responses, just respond to the human's message in the \
context of the conversation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)


# Callbacks support token-wise streaming for debugging
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llama3_inst_path = os.environ["LLAMA3_PATH"]

llm = LlamaCpp(
    model_path=llama3_inst_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    # callback_manager=callback_manager,
    # verbose=True,  # Verbose is required to pass to the callback manager
    verbose=False,
    top_p=0.0001,
    max_tokens=2000,
    n_ctx=2048,
)

llm_chain = prompt | llm

# TODO: incorporate RAG for user info and past convos - use langchain chroma interface??
def chat_session():
  session_id = str(uuid4())
  chat_histories = {session_id: ChatMessageHistory()}
  chain_with_message_history = RunnableWithMessageHistory(
      llm_chain,
      lambda session_id: chat_histories[session_id],
      input_messages_key="input",
      history_messages_key="chat_history",
  )

  while True:
    user_input = str(input("User: "))

    # TODO: fix Parent run <id> not found for run <diff-id>. Treating as a root run
    response = chain_with_message_history.invoke(
        # add EOS token after user input to not complete it
        {"input": user_input + "<|eot_id|>"},
        {"configurable": {"session_id": session_id}},
    )

    print(response)

chat_session()