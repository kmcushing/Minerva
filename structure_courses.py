
from pdfminer.high_level import extract_text
import re
import json
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
course_data= extract_text("Course_DB.pdf").splitlines()


def parse_courses(text):
    #Needs further checks for non-course titles - (next to do)
    # Regex pattern to capture title + descriptions
    course_pattern = r"([A-Za-z\s]+ (?:Major|Minor|Degree))(.*?)(?=\n[A-Za-z\s]+ (?:Major|Minor|Degree)|$)"
    courses = []
    titles = []
    num= -1
    #print(text)
    # Find all courses in the text
    for line in text:
        description= True
        for match in re.finditer(course_pattern, line, re.DOTALL | re.MULTILINE):
            title =  match.group(1).strip()
            description = match.group(2).strip()
            if "The " not in title:
                title = "The " + title
            if titles:
                #checks if a title exists inside a used title
                title_words = set(title.split())
                dist = max([title_words.intersection(set(word.split())) for word in titles])
                if len(dist) == len(title.split()):
                    break
            if title not in titles:
                # Structure the course data
                course_info = {
                    "Course Title": title,
                    "Description": ''
                }
                courses.append(course_info)
                titles.append(title)
                curr = title
                num += 1
                description= False
        if description:
            courses[num]["Description"] +=  '\n'+ line
        
    return courses


def main():
    course_data= parse_courses(extract_text("Course_DB.pdf").splitlines())
    # Load & Write course data to a JSON file
    with open('courses.json', 'w') as json_file:
        json.dump(course_data, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()