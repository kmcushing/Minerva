import chromadb
from datetime import datetime
from uuid import uuid4
import os
import traceback

DATE_FORMAT = "%m-%d-%YT%H:%M:%S"

os.environ["CHAT_STORAGE_PATH"] = 'data/chats'
client = chromadb.PersistentClient(path=os.environ["CHAT_STORAGE_PATH"])


# TODO: fix "Context leak detected" message
def store_user_message(username, input, response):
    collection = client.get_or_create_collection(username)
    # add embeddings for both user message and response
    collection.add(
        documents=input,
        metadatas={
            "timestamp": datetime.now().strftime(DATE_FORMAT),
            "role": "user",
            "paired_message": response,
        },
        ids=str(uuid4()),
    )
    collection.add(
        documents=response,
        metadatas={
            "timestamp": datetime.now().strftime(DATE_FORMAT),
            "role": "minerva",
            "paired_message": input,
        },
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

    # assuming only one query - sort by recency
    data = sorted(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        key=lambda x: datetime.strptime(x[1]["timestamp"], DATE_FORMAT),
        reverse=True,
    )
    # print(data)

    relevant_messages = []
    for message, metadata, dist in data:
        if dist <= max_distance:
            relevant_messages.append(
                (
                    message,
                    metadata["role"],
                    metadata["paired_message"],
                )
            )

    return relevant_messages


def retrieve_and_format_user_messages(
    username,
    query,
    max_results=10,
    max_distance=2.0,
):
    try:
        relevant_messages = retrieve_user_messages(
            username, query, max_results, max_distance
        )
        pairs = []
        for message, role, paired_message in relevant_messages:
            pair = tuple(
                (message, paired_message)
                if role == "user"
                else (paired_message, message)
            )
            # print(pair)
            if pair not in pairs:
                pairs.append(pair)
        if not pairs:
            return ""
        s = "Relevant Past Massages:\n"
        for user_message, minerva_message in pairs:
            s += f"User: {user_message}\nMinerva: {minerva_message}\n"
        return s
    except Exception as e:
        # print(e)
        # print(traceback.format_exc())
        return ""


# INITIAL TESTING

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

# ms = retrieve_and_format_user_messages(
#     "user5",
#     "remind me of the steps needed to create an ai model for time-series forecasting",
# )

# print(ms)
