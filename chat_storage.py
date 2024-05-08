import chromadb
from datetime import datetime
from uuid import uuid4

STORAGE_PATH = "data/chat_embeddings"
DATE_FORMAT = "%m-%d-%YT%H:%M:%S"

client = chromadb.PersistentClient(path=STORAGE_PATH)


# TODO: fix "Context leak detected" message
def store_user_message(username, input, response):
    collection = client.get_or_create_collection(username)
    collection.add(
        documents=f"User: {input}\nAssistant: {response}",
        metadatas={"timestamp": datetime.now().strftime(DATE_FORMAT)},
        ids=str(uuid4()),
    )
    return True


# TODO: add filtering by a date or length of time from datetime.now()
def retrieve_user_messages(username, query, max_results=10, max_distance=2.0):
    # assumes collection for user exists
    collection = client.get_collection(username)
    results = collection.query(
        query_texts=query,
        n_results=max_results,
    )

    # assuming only one query - sort by dist
    data = sorted(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        key=lambda x: x[-1],
    )

    relevant_messages = []
    for doc, date, dist in data:
        if dist <= max_distance:
            relevant_messages.append(doc)

    return relevant_messages


# store_user_message(
#     "test_user_1",
#     "I like Computer Science.",
#     "That's great! I will try to remember that for our future conversations.",
# )
# store_user_message(
#     "test_user_1",
#     "I am a Math major.",
#     "That's great! I will try to remember that for our future conversations.",
# )

# ms = retrieve_user_messages("test_user_1", "algorithm")

# print(ms)
