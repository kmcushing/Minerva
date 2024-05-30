import getpass
import os
from transformers import pipeline
from user_representation import User, load_user
from course_info_storage import COURSE_COLLECTION, retrieve_and_format_courses
from chat_storage import store_user_message, retrieve_and_format_user_messages
import random
import google.generativeai as genai


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)


DM_SYSTEM_PROMPT = """\
You're job is to extract the Topic, Insider Knowledge, and Violations from a given message from a user as specified here. \
Please begin your response with "Topic:", followed by the general topic of the user's previous message. 
Keep this topic as brief and general as possible while still accurately capturing the topic of the message. \
If the user is asking about or referring to courses at Northwestern, the topic should be "Northwestern Courses". \
Follow this with a new line and "Insider Knowledge:", followed by the specific domains that the user's last message conveys insider knowledge of. 
If there are multiple such domains, output each one separated by a comma. \
If the user's message only conveys outsider knowledge of domains, output "None". \
Follow this with a new line and "Violations:", and using the provided Gricean maxim violation examples provides, determine if the user's response indicates a violation in the prior turn of the conversation for each maxim of "Quantity", "Quality", "Relation", or "Manner". \
If there are multiple such violations, output each one separated by a comma. \
The Gricean maxims are defined as follows: 
Quantity: where one tries to be as informative as one possibly can, and gives as much information as is needed, and no more
Quality: where one tries to be truthful, and does not give information that is false or that is not supported by evidence
Relation: where one tries to be relevant, and says things that are pertinent to the discussion
Manner: when one tries to be as clear, as brief, and as orderly as one can in what one says, and where one avoids obscurity and ambiguity

Some example phrases that can indicate violations of each maxim are as follows:
{maxim_violation_examples}
{user_info}
"""

MINERVA_SYSTEM_PROMPT = """\
You are Minerva, an Academic Advisor conversational assistant at Northwestern University, skilled at crafting responses to \
effectively communcate with a specific user given some information about them. \
Do NOT generate human responses, just respond to the human's message in the \
context of the conversation. Using the user profile and parameters of Gricean maxims, cater your response to the user. You are not a student! \
If the user has insider knowledge of a domain, you can assume the user knows terms within the domain without thorougly explaining them. Overexplaining violates the Gricean maxim of Quantity. \
If the user does not have insider knowledge of a domain, please explain any domain-specific terms used in a way that they would understand. Underexplaining also violates the Gricean maxim of Quantity. \
If possible, try to provide analogies for concepts in domains where the user does not have insider knowledge to concepts in domains that the user has insider knowledge of. \
Begin all responses with "Minerva:". When answering questions about a specifc class, only use information that you have previously provided \
and do not make up any new information. Do not mention anything about the professor teaching the class as that information is not available.
{maxim_info}
{user_info}
{context}
"""

# adapted from https://www.sas.upenn.edu/~haroldfs/dravling/grice.html#:~:text=The%20maxim%20of%20quantity%2C%20where,is%20not%20supported%20by%20evidence
MINERVA_MAXIMS_DESCRIPTION = """Please adhere to the following Gricean maxims for conversation.
Quantity: where one tries to be as informative as one possibly can, and gives as much information as is needed, and no more
Quality: where one tries to be truthful, and does not give information that is false or that is not supported by evidence
Relation: where one tries to be relevant, and says things that are pertinent to the discussion
Manner: when one tries to be as clear, as brief, and as orderly as one can in what one says, and where one avoids obscurity and ambiguity"""


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


def maxim_violations_to_string():
    s = ""
    for m in maxims:
        s += f"The following phrases cna indicate violations of {m}:\n"
        for p in maxims[m]:
            s += f"{p}\n"
    return s


class DialogueManager:
    def __init__(self, user, llm):
        self.violation = {"quantity": 1, "quality": 1, "relation": 1, "manner": 1}
        self.num_turns = 0
        # self.classify = spacy.load("en_core_web_sm")
        # only need to update user rep in this for each query
        self.dm_chat = llm.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": "System Prompt: "
                    + DM_SYSTEM_PROMPT.format(
                        maxim_violation_examples=maxim_violations_to_string(),
                        user_info=user.to_prompt(),
                    ),
                },
                {
                    "role": "model",
                    "parts": "Understood.",
                },
            ]
        )
        # initially retrieve some past messages about a random topic the user has discussed
        dit = user.topic_frequencies
        # choose a random topic to retrieve info - may want to sample based on topic frequencies
        context = (
            retrieve_and_format_user_messages(
                user.id,
                list(dit.keys())[random.randint(0, len(dit.keys()) - 1)],
            )
            if dit
            else None
        )
        self.minerva_chat = llm.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": "System Prompt: "
                    + MINERVA_SYSTEM_PROMPT.format(
                        maxim_info=MINERVA_MAXIMS_DESCRIPTION,
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
        self.user = user
        self.course_info_idx = 1

    def update_gricean_atts(self, prompt, violations):
        self.num_turns += 1
        sent = classifier(prompt)
        label, score = (sent[0]["label"], sent[0]["score"])
        for v in self.violation:
            if v in violations:
                self.violation[v] -= 0.02 * score
            else:
                self.violation[v] += 0.01 * score
            if self.violation[v] > 1:
                self.violation[v] = 1
            elif self.violation[v] < -1:
                self.violation[v] = -1
        return self.violation

    def retrieve_and_insert_courses(self, user_input):
        self.minerva_chat.history.insert(
            self.course_info_idx,
            {"role": "user", "parts": ""},
        )
        self.course_info_idx += 1
        self.minerva_chat.history.insert(
            self.course_info_idx,
            {
                "role": "model",
                "parts": retrieve_and_format_courses(user_input),
            },
        )
        self.course_info_idx += 1

    def dm_invoke(self, user_input):
        # generate response
        response = self.dm_chat.send_message(user_input).text
        if (
            "Topic:" not in response
            or "Insider Knowledge:" not in response
            or "Violations: " not in response
        ):
            print("ERROR: response format incorrect for response", response)
            return None
        topic_start = response.index("Topic: ") + len("Topic: ")
        topic_end = response.index("Insider Knowledge: ")
        topic = response[topic_start:topic_end].strip().lower()
        domain_start = topic_end + len("Insider Knowledge: ")
        violation_start = response.index("Violations: ") + len("Violations: ")
        domains = [
            d.strip().lower()
            for d in str(response[domain_start : response.index("Violations:")]).split(
                ","
            )
        ]
        # update system violations
        violations = [
            v.strip().lower()
            for v in str(response[violation_start:]).split(",")
            if v.strip().lower() != "none"
        ]
        self.update_gricean_atts(user_input, violations)
        # update user's topics
        self.user.discussed_topic(topic)
        # if message is about courses, retrieve and aoppend course info to chat history
        if "course" in topic.lower() or "class" in topic.lower():
            self.retrieve_and_insert_courses(user_input)
        # update user's insider domains
        [self.user.add_insider_domain(d) for d in domains if d != "none"]
        return violations

    def handle_user_message(self, user_input):
        # first invoke dm
        violations = self.dm_invoke(user_input)
        # then use dm to update chat history
        violation_str = (
            ""
            if not violations
            else " You may have violated the following maxims in your previous response: "
            + " ".join(violations)
        )
        # UNCOMMENT THE FOLLOWING 2 LINES IF YOU WANT TO PRINT OUT DETECTED VIOLATIONS
        # if violations:
        # print("DETECTED POSSIBLE VIOLATIONS:", violations)

        context = retrieve_and_format_user_messages(self.user.id, user_input)
        self.minerva_chat.history[0].parts[0].text = (
            "System Prompt: "
            + MINERVA_SYSTEM_PROMPT.format(
                maxim_info=MINERVA_MAXIMS_DESCRIPTION + violation_str,
                user_info=self.user.to_prompt(),
                context=context,
            )
        )
        response = self.minerva_chat.send_message(user_input).text
        store_user_message(self.user.id, user_input, response)
        return response


def chat_session(test=False):

    if not test:
        # user_id cant be same as the name of our collection for course info
        invalid_user_id = True
        while invalid_user_id:
            user_id = input("Please enter your username: ")
            if user_id != COURSE_COLLECTION:
                invalid_user_id = False
        user = load_user(user_id)

        llm = genai.GenerativeModel("gemini-1.5-pro-latest")

        dialouge_manager = DialogueManager(user, llm)

        welcome = dialouge_manager.minerva_chat.send_message(
            "Craft a welcome message for the user based on your previous conversation."
        )
        print(welcome.text)

        while True:
            user_input = input("You: ")
            if user_input == "exit":
                break
            response = dialouge_manager.handle_user_message(user_input)
            print(response)
    # TODO: refactor this to use the updated DM class
    # else:
    #     user_info = "test"
    #     llm = genai.GenerativeModel("gemini-pro")
    #     dialouge_manager = Prompt_Handler(llm, user_info)
    #     # set initial chat history with system prompt - update history[0] to reflect most up to date infor about the user
    #     chat = llm.start_chat(
    #         history=[
    #             {
    #                 "role": "user",
    #                 "parts": "System Prompt: "
    #                 + SYSTEM_PROMPT.format(
    #                     user_info=None,
    #                     relevant_messages=None,
    #                 ),
    #             },
    #             {
    #                 "role": "model",
    #                 "parts": "Understood.",
    #             },
    #         ]
    #     )

    #     welcome = chat.send_message(
    #         f"Craft a welcome message for the user based on your previous conversation. For this message, only output your response."
    #     )
    #     print(welcome.text)
    #     for msg in test:
    #         param = dialouge_manager.gricean_att(msg)
    #         response = chat.send_message(msg)
    #         # response = chat.send_message("Model guides: " + str(param) + "\n" + "User: " + msg)
    #         print("User: " + msg)
    #         print(response.text)
    #     print(param)


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

# uncomment for testing
# test_cases = [potential_cs_student, potential_econ_student, senior_transfer_advisee]

# for c in test_cases:
#     chat_session(c)
#     print(
#         "-----------------------------------------------------------------------------------------"
#     )

chat_session()
