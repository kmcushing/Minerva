import getpass
import os
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from transformers import pipeline
import spacy 
#import chromadb


#client = chromadb.PersistentClient(path="/path/to/save/to")
#
#
#collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
#collection = client.get_collection(name="my_collection", embedding_function=emb_fn)


#doc = nlp(content)
 
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
                        pass
                else:
                    self.violation[mx] += .1 * score
        else:
            pass
            #test for sensitivity

        #parase through prompt and assign new gricean values
        return self.violation,self.num_turns
    def domain_builder(self,prompt):
        doc = self.nlp(prompt)
        # Extract and categorize entities dynamically
        for ent in doc.ents:
            #self.persona[ent.label_]["Entities"].append(ent.text)
            self.persona[ent.text]["Counts"][ent.text] += 1

    
#docs = 

'''

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
'''

#db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
#docs = db.similarity_search(query)

user_info= input('Please enter your username')
#print welcome message for user(first time/returning)
#create messeage based on user profile, if user profile is none = print generic welcome 
model = Prompt_Handler()
while True:
    query = input()
    results= model.gricean_att(query)
    print(results)
    #result = llm.invoke("Write a ballad about LangChain")
    #print(result.content)
    pass
