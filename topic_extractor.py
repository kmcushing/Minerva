from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="valurank/distilroberta-topic-classification",
    device="mps",
)


def extract_topic(statement):
    return pipe(statement)[0]["label"]


# print(extract_topic("Hello"))
