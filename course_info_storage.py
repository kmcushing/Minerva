import chromadb
from datetime import datetime
from uuid import uuid4
import os
import traceback
import re

DATE_FORMAT = "%m-%d-%YT%H:%M:%S"
os.environ["COURSES_STORAGE_PATH"] = 'data/chroma'
client = chromadb.PersistentClient(path=os.environ["COURSES_STORAGE_PATH"])

COURSE_COLLECTION = "course_information"


def store_course_info(title, description, extra_info):
    collection = client.get_or_create_collection(COURSE_COLLECTION)
    # add course to collection - use title for embedding if no description available
    collection.add(
        documents=description if description else title,
        metadatas={
            "title": title,
            "extra_info": extra_info,
        },
        ids=str(uuid4()),
    )
    return True


def retrieve_relevant_courses(query, max_results=5, max_distance=2.0):
    # assumes collection for user exists
    collection = client.get_collection(COURSE_COLLECTION)
    results = collection.query(
        query_texts=query,
        n_results=max_results,
    )

    # assuming only one query - default sort by distance
    data = zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )
    # print(data)

    relevant_courses = []
    for doc, metadata, dist in data:
        if dist <= max_distance:
            relevant_courses.append(
                (
                    doc,
                    metadata["title"],
                    metadata["extra_info"],
                )
            )

    return relevant_courses


def retrieve_and_format_courses(
    query,
    max_results=10,
    max_distance=2.0,
):
    try:
        relevant_courses = retrieve_relevant_courses(
            query,
            max_results,
            max_distance,
        )
        s = "Relevant Courses:\n"
        for description, title, extra_info in relevant_courses:
            s += f"Course Title: {title}"
            s += f" Description: {description}" if description != title else ""
            s += f" Additional Info: {extra_info}\n" if extra_info else "\n"
        return s
    except Exception as e:
        # print(e)
        # print(traceback.format_exc())
        return ""


# INITIAL TESTING

# ms = retrieve_and_format_courses("I like ai")

# print(ms)
