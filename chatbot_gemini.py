import google.generativeai as genai
from uuid import uuid4
import os
from user_representation import User, load_user
from topic_extractor import extract_topic

# TODO: take this in from CLI
USER_ID = "user2"
USER_REPRESENTATION = load_user(USER_ID)

# setting up prompt with chat history
SYSTEM_PROMPT = """\
You are Minerva, a conversational assistant, skilled at crafting responses to \
effectively communcate with a specific user given some information about them. \
Do NOT generate human responses, just respond to the human's message in the \
context of the conversation.
{user_prompt}"""

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_PROMPT),
#         MessagesPlaceholder("chat_history", optional=True),
#         ("human", "{input}"),
#     ]
# )


# Callbacks support token-wise streaming for debugging
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# n_gpu_layers = -1
# n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# # Make sure the model path is correct for your system!
# llama3_inst_path = os.environ["LLAMA3_PATH"]

# llm = LlamaCpp(
#     model_path=llama3_inst_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     # callback_manager=callback_manager,
#     # verbose=True,  # Verbose is required to pass to the callback manager
#     verbose=False,
#     # top_p=0.0001,
#     max_tokens=2000,
#     n_ctx=2048,
# )

# llm_chain = prompt | llm


# TODO: incorporate RAG for user info and past convos - use langchain chroma interface??
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def chat_session():
    session_id = str(uuid4())
    model = genai.GenerativeModel(
        "gemini-pro",
    )
    chat = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": "System Prompt: "
                + SYSTEM_PROMPT.format(user_prompt=USER_REPRESENTATION.to_prompt()),
            },
            {
                "role": "model",
                "parts": "Understood.",
            },
        ]
    )

    while True:
        user_input = str(input("User: "))
        # update topics for user
        # TODO: exclude greatings and things with minimal content
        USER_REPRESENTATION.disscussed_topic(extract_topic(user_input))
        # TODO: determine if input demonstrates insider knowledge - maybe ask gemini to return that in response somehow?

        # update system prompt to reflect most recent info about the user
        chat.history[0].parts[0].text = "System Prompt: " + SYSTEM_PROMPT.format(
            user_prompt=USER_REPRESENTATION.to_prompt()
        )
        response = chat.send_message(
            user_input,
        )
        print(chat.history)
        print(response.text)


chat_session()
