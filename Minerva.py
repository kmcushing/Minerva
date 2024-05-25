import getpass
import os
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from transformers import pipeline
import spacy
from user_representation import User, load_user
from course_info_storage import COURSE_COLLECTION, retrieve_and_format_courses
from chat_storage import store_user_message, retrieve_and_format_user_messages
import random
import json

# using googles api instead of langchain for direct access to update system prompt
import google.generativeai as genai
from google.ai.generativelanguage import Content


path = "data/courses.json"

# Open the JSON file and load its content into a variable
with open(path, "r") as file:
    data = json.load(file)

# read catalog
file_path = "data/catalog.txt"
with open(file_path, "r") as file:
    catalog = file.read()


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

SYSTEM_PROMPT = """\
You are Minerva, an Academic Advisor conversational assistant at Northwestern University, skilled at crafting responses to \
effectively communcate with a specific user given some information about them. \
Do NOT generate human responses, just respond to the human's message in the \
context of the conversation. Using the user profile, department data, and parameters of Gricean maxims, cater your response to the user. You are not a student! \
If the user has insider knowledge of a domain, you can assume the user knows terms within the domain without thorougly explaining them. \
If the user does not have insider knowledge of a domain, please explain any domain-specific terms used in a way that they would understand. \
If possible, try to provide analogies for concepts in domains where the user does not have insider knowledge to concepts in domains that the user has insider knowledge of. \
Please begin your response with "Topic:", followed by the general topic of the user's previous message. 
Keep this topic as brief and general as possible while still accurately capturing the topic of the message. \
Follow this with a new line and "Insider Knowledge:", followed by the specific domains that the user's last message conveys insider knowledge of. 
If there are multiple such domains, output each one separated by a comma. \
If the user's message only conveys outsider knowledge of domains, output "None". \
Follow this with your response to the user's message. Begin all responses with "Minerva:". \
When answering questions about a specifc class, only use information that you have previously provided \
and do not make up any new information.
{user_info}
{context}
"""

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-pro")

# for ent in doc.ents:
# print(ent.text, ent.start_char, ent.end_char, ent.label_)


# Load a pre-trained sentiment-analysis pipeline, specify a model
classifier = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


maxims = {
    "Quantity": [
        "What else",
        "Anything more",
        "Is that all",
        "Can you add more",
        "Is there more to it",
        "Tell me more",
        "Is there anything else",
        "What more",
        "More details please",
    ],
    "Quality": [
        "That's incorrect",
        "I didn't say that",
        "That's not true",
        "You're wrong",
        "That's false",
        "I disagree",
        "That's a lie",
        "That's misleading",
        "Not accurate",
        "That's a mistake",
        "That's an error",
        "Not true",
        "False information",
    ],
    "Relation": [
        "Unrelated information",
        "off-topic",
        "That's irrelevant",
        "Not related",
        "What does that have to do with this?",
        "How is that connected?",
        "That's beside the point",
        "Not the point",
        "Irrelevant",
        "That's not the issue",
        "That's off the subject",
        "Off track",
    ],
    "Manner": [
        "What are you talking about",
        "What do you mean",
        "I'm lost",
        "huh",
        "I don't understand",
        "Can you clarify?",
        "That's confusing",
        "Please explain",
        "I don't get it",
        "Pardon",
        "Can you elaborate?",
        "Not clear",
        "Can you rephrase?",
        "That's unclear",
        "I'm confused",
        "Can you simplify",
        "Can you be more specific?",
        "What does that mean",
        "Ambiguous",
        "Not sure what you mean",
        "unclear",
    ],
}


class Prompt_Handler:
    def __init__(self, llm, user):
        self.violation = {"Quantity": 0, "Quality": 0, "Relation": 0, "Manner": 0}
        self.num_turns = 0
        self.classify = spacy.load("en_core_web_sm")
        self.user_domains = {}
        self.llm = llm
        self.user = user

    def gricean_att(self, prompt):
        self.num_turns += 1
        sent = classifier(prompt)
        label, score = (sent[0]["label"], sent[0]["score"])
        if label == "NEGATIVE":
            for mx, phrase in maxims.items():
                if any(ph.lower() in prompt.lower() for ph in phrase):
                    self.violation[mx] -= 0.2 * score
                    if mx == "Quality":
                        # prompt to ask the user for more information and update records
                        result = self.llm.invoke(
                            f"{SYSTEM_PROMPT+'Ask the user for more information on what is incorrect'}"
                        )
                        print(result.content)
                        prompted = True
                else:
                    self.violation[mx] += 0.1 * score
        else:
            for mx, phrase in maxims.items():
                self.violation[mx] += 0.1 * score
            # test for sensitivity
        # parase through prompt and assign new gricean values
        return self.violation

    def domain_builder(self, prompt):
        doc = self.classify(prompt)
        # Extract and categorize entities dynamically
        for ent in doc.ents:
            self.user.disscussed_topic(ent.text)

    def response_handler(self, user_input, response):
        if (
            "Minerva:" not in response
            or "Topic:" not in response
            or "Insider Knowledge:" not in response
        ):
            print("ERROR: response format incorrect for response", response)
            return None
        topic_start = response.index("Topic: ") + len("Topic: ")
        topic_end = response.index("Insider Knowledge: ")
        topic = response[topic_start:topic_end].strip().lower()
        domain_start = topic_end + len("Insider Knowledge: ")
        message_start = response.index("Minerva: ")
        domains = [
            d.strip().lower()
            for d in str(response[domain_start:message_start]).split(",")
        ]
        message = response[message_start:].strip()
        # update user's topics
        self.user.discussed_topic(topic)
        # update user's insider domains
        [self.user.add_insider_domain(d) for d in domains if d != "none"]
        # update user's messages
        store_user_message(self.user.id, user_input, message)
        return message


def chat_session(test=False):

    if not test:
        # user_id cant be same as the name of our collection for course info
        invalid_user_id = True
        while invalid_user_id:
            user_id = input("Please enter your username: ")
            if user_id != COURSE_COLLECTION:
                invalid_user_id = False
        user = load_user(user_id)

        dit = user.topic_frequencies
        # choose a random topic to retrieve info - may want to sample based on topic frequencies
        context = (
            retrieve_and_format_user_messages(
                user_id,
                list(dit.keys())[random.randint(0, len(dit.keys()) - 1)],
            )
            if dit
            else None
        )
        # print(context)
        llm = genai.GenerativeModel("gemini-pro")
        # set initial chat history with system prompt - update history[0] to reflect most up to date infor about the user
        chat = llm.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": "System Prompt: "
                    + SYSTEM_PROMPT.format(
                        user_info=user.to_prompt(),
                        context=context,
                    ),
                },
                {
                    "role": "model",
                    "parts": "Understood.",
                },
            ]
        )

        welcome = chat.send_message(
            f"Craft a welcome message for the user based on your previous conversation. For this message, only output your response."
        )
        print(welcome.text)
        dialouge_manager = Prompt_Handler(llm, user)
        prompted = False
        course_info_idx = 1

        while True:
            user_input = input("You: ")
            if user_input == "exit":
                break
            # update system prompt with more recent info and updated relevant sources
            context = retrieve_and_format_user_messages(user_id, user_input)
            print("CONTEXT:", context)
            chat.history[0].parts[0].text = "System Prompt: " + SYSTEM_PROMPT.format(
                user_info=user.to_prompt(),
                context=context,
            )
            chat.history.insert(
                course_info_idx,
                {"role": "user", "parts": ""},
            )
            course_info_idx += 1
            chat.history.insert(
                course_info_idx,
                {
                    "role": "model",
                    "parts": "Topic: None\nInsider Knowledge: None\nMinerva:"
                    + retrieve_and_format_courses(user_input),
                },
            )
            course_info_idx += 1
            # want to append classes to history rather than system prompt since
            # future user responses may reference the class without context
            # that would be required to retrieve it again

            print("USER INFO:", user.to_prompt())
            # param = dialouge_manager.gricean_att(user_input)

            response = chat.send_message(user_input)
            # print(result.content)
            # store_user_message(user_id, user_id, response.text)
            # print(response.text)
            output = dialouge_manager.response_handler(user_input, response.text)
            print(output)
            prompted = False
    else:
        user_info = "test"
        llm = genai.GenerativeModel("gemini-pro")
        dialouge_manager = Prompt_Handler(llm, user_info)
        # set initial chat history with system prompt - update history[0] to reflect most up to date infor about the user
        chat = llm.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": "System Prompt: "
                    + SYSTEM_PROMPT.format(
                        user_info=None,
                        relevant_messages=None,
                    ),
                },
                {
                    "role": "model",
                    "parts": "Understood.",
                },
            ]
        )

        welcome = chat.send_message(
            f"Craft a welcome message for the user based on your previous conversation. For this message, only output your response."
        )
        print(welcome.text)
        for msg in test:
            param = dialouge_manager.gricean_att(msg)
            response = chat.send_message(msg)
            # response = chat.send_message("Model guides: " + str(param) + "\n" + "User: " + msg)
            print("User: " + msg)
            print(response.text)
        print(param)


# chat_session()


potential_cs_student = [
    "Hi Minerva, my name is Dave! I am a first-year student here at Northwestern. I am from New York and I went to high school at RPT. I've always been interested in technology and how it can be used creatively. I’ve taken a few math courses but I didn’t do too well in them so not sure what field I can go into without liking math.",
    "Yes! I'm excited to be here as well! Looking forward to deciding on a major here at Northwestern. Can you suggest a major based on my skills?",
    " What is that major? I've never heard of that!",
    "Oh, wow. Tell me more.",
    "I love experimenting with Generative AI and creating unique digital art.",
    "Can you explain how computer science relates to math? I don't get it?",
    "Is there a way to combine my passion for creative technology with computer science?",
    "Can you rephrase that?",
    "What kind of projects can I work on in computer science that involve creativity?",
    "Are there any introductory resources for computer science that you recommend?What course load would you recommend for me?",
    "exit",
]

# Potential Econ Student(With insider Knowledge)

potential_econ_student = [
    "Hi Minerva, my name is Mo! I am a first-year student here at Northwestern. I am from Virginia and I went to high school at WestBerry]. A favorite project I have done involved studying how the law of diminishing marginal returns is shown in production in my home city. I also studied utility maximization in local small businesses around my school",
    "Yes! I'm excited to be here as well! Looking forward to deciding on a major here at Northwestern.",
    "What is that major? I've never heard of that!",
    "My father is a consultant, and he always talks about how the economy plays a crucial role in his job.",
    "I took calculus in high school and found it fascinating, especially the applications in economics.",
    "Do you think my background in calculus will help me with the quantitative aspects of economics?",
    "I'm curious about how economic theories can be applied to real-world consulting.",
    "Can you be more specific?",
    "What are the key skills I need to succeed in an economics program?",
    "Oh, okay sounds good.",
    "Is that all?",
    "I'm looking forward to learning more in class",
    "exit",
]

# Senior Transfer Advisee (missing a credit from required course list but isn’t aware)

senior_transfer_advisee = [
    "Hi Minerva, my name is too_lazy! I am a senior transfer student here at Northwestern. I am from Ohio and I went to high school at Rando.",
    "Sort of exhausted. Looking forward to graduating soon though!",
    "Were you able to get my transcript and notes from my previous advisor",
    "Oh really? Thank you!",
    "Is there anything else?",
    "exit",
]

# test_cases = [potential_cs_student, potential_econ_student, senior_transfer_advisee]

# for c in test_cases:
#     chat_session(c)
#     print(
#         "-----------------------------------------------------------------------------------------"
#     )

chat_session()
