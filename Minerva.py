import getpass
import os
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from transformers import pipeline
import spacy 
from user_representation import User, load_user

from chat_storage import store_user_message, retrieve_user_messages
import random



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")



user_info= input('Please enter your username')
curr_user = load_user(user_info)
 
#for ent in doc.ents:
    #print(ent.text, ent.start_char, ent.end_char, ent.label_)



# Load a pre-trained sentiment-analysis pipeline, specify a model
classifier = pipeline('sentiment-analysis',model = "distilbert-base-uncased-finetuned-sst-2-english")


maxims = {
    "Quantity": [
        "What else", "Anything more", "Is that all", "Can you add more", "Is there more to it", 
        "Tell me more", "Is there anything else", "What more", "More details please"
    ],
    "Quality": [
        "That's incorrect", "I didn't say that", "That's not true", "You're wrong", "That's false", 
        "I disagree", "That's a lie", "That's misleading", "Not accurate", "That's a mistake",
        "That's an error", "Not true", "False information"
    ],
    "Relation": [
        "Unrelated information", "off-topic", "That's irrelevant", "Not related", "What does that have to do with this?", 
        "How is that connected?", "That's beside the point", "Not the point", "Irrelevant", 
        "That's not the issue", "That's off the subject", "Off track"
    ],
    "Manner": [
        "What are you talking about", "What do you mean", "I'm lost", "huh", "I don't understand", 
        "Can you clarify?", "That's confusing", "Please explain", "I don't get it", "Pardon", 
        "Can you elaborate?", "Not clear", "Can you rephrase?", "That's unclear", "I'm confused", 
        "Can you simplify", "Can you be more specific?", "What does that mean", "Ambiguous", 
        "Not sure what you mean", "unclear"
    ]
}

class Prompt_Handler:
    def __init__(self) -> None:
        self.violation = { 'Quantity' : 0, 'Quality' : 0, 'Relation': 0, 'Manner' : 0}
        self.num_turns = 0
        self.classify= spacy.load("en_core_web_sm")
        self.user_domains = {}

    def normalize(self,prompt):
        pass

    def gricean_att(self,prompt):
        self.num_turns += 1
        sent = classifier(prompt)
        label,score = (sent[0]['label'],sent[0]['score'])
        if label == "NEGATIVE":
            for mx, phrase in maxims.items():
                if any(ph.lower() in prompt.lower() for ph in phrase):
                    self.violation[mx] -= .2 * score
                    if mx == "Quality":
                        #prompt to ask the user for more information and update records
                        result = llm.invoke(f"{SYSTEM_PROMPT+'Ask the user for more information on what is incorrect'}")
                        print(result.content)
                        prompted = True
                else:
                    self.violation[mx] += .1 * score
        else:
            for mx, phrase in maxims.items():
                self.violation[mx] += .1 * score
            #test for sensitivity

        #parase through prompt and assign new gricean values
        return self.violation,self.num_turns
    def domain_builder(self,prompt):
        doc = self.classify(prompt)
        # Extract and categorize entities dynamically
        for ent in doc.ents:
            curr_user.disscussed_topic(ent.text)
            

    





SYSTEM_PROMPT = """\
You are Minerva, an Academic Advisor conversational assistant at Northwestern University, skilled at crafting responses to \
effectively communcate with a specific user given some information about them. \
Do NOT generate human responses, just respond to the human's message in the \
context of the conversation.  Using the user profile and parameters of Gricean maxims, cater your response to the user.You are not a student!"""


dit= curr_user.topic_frequencies
context = retrieve_user_messages(user_info,list(dit.keys())[random.randint(0,2)])

welcome = llm.invoke(f"{SYSTEM_PROMPT + 'Craft a welcome message for the user based on your previous conversation' + user_info + str(curr_user) + str(context)}")
print(welcome.content)
model = Prompt_Handler()
prompted= False

while True:
    query = input()
    if query == 'exit':
        break
    results= model.gricean_att(query)
    model.domain_builder(query)
    if not prompted:
        result = llm.invoke(f"{SYSTEM_PROMPT + str(maxims) + query}")
        print(result.content)
    store_user_message(user_info,query,result.content)
    prompted = False
