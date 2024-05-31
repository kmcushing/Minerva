# Minerva

Minerva is a chat-bot system that is designed to act as a Northwestern academic advisor, although it can have conversations about any topic. Minerva learns a representation of a user comprised of their frequently discussed topics and domains that they have insider knowledge of to more effective communicate with users.

## Setup

Create a local virtual environment with Python 3.10 (can be done by running `python3.10 -m venv [your-venv-name]`). Activate your virtual environment by running `source [your-venv-name]/bin/activate` and install the requirements from `requirements.txt` by running `pip install -r requirements.txt`. Define the following environment variables however you like \(we recommend writing these to a file named `.envrc` with each variable and value prefaced by `export` so that you can define them by running this file each time you start a new terminal session\):

- `CHAT_STORAGE_PATH` set to the relative path to a directory where you would like to store the persistent ChromaDB collections for conversation histories
- `COURSES_STORAGE_PATH` set to the relative path to a directory where you would like to store the persistent ChromaDB collection for courses and their descriptions
- `GOOGLE_API_KEY` set to the value of an API key for Gemini - you can create this key following the docs [here](https://ai.google.dev/gemini-api/docs/api-key) - if you do not have a GCP account you will have to create one

After defining your enviornment variables, you will need to extract course information from the Northwestern course catalog webpage by running `python extract_course_info.py` - this will take several minutes. Only do this the first time you set up this repo, otherwise courses will be duplicated in the corresponding ChromaDB collection, leading to the same course occuring multiple times in the prompt for Gemini.

## Using Minerva

Inside of your virtual environment, after installing the requirements, ensure your environment variables are defined and run `python Minerva.py` from the root directory of this folder. You will be prompted to input a username and then you can chat with Minerva.
