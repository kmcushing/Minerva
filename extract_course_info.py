# modified from ChatGPT response to save to chroma collection
# and add addtitional parsing for typos in html classes on website
import requests
from bs4 import BeautifulSoup
from course_info_storage import store_course_info


def get_links_from_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    az_sitemap_div = soup.find("div", class_="az_sitemap")
    links = [a["href"] for a in az_sitemap_div.find_all("a", href=True)]
    return links


def extract_courseblock_text(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    courseblocks = soup.find_all("div", class_="courseblock")

    courses_info = []
    for courseblock in courseblocks:
        title = courseblock.find("span", class_="courseblocktitle")
        if not title:
            title = courseblock.find("p", class_="courseblocktitle")
        if not title:
            title = courseblock.find("span", class_="couresblocktitle")
        if not title:
            title = courseblock.find("p", class_="couresblocktitle")
        desc = courseblock.find("span", class_="courseblockdesc")
        if not desc:
            desc = courseblock.find("p", class_="courseblockdesc")
        extra_spans = courseblock.find_all("span", class_="courseblockextra")
        extra_spans += courseblock.find_all("p", class_="courseblockextra")

        course_info = tuple(
            (
                title.get_text(separator=" ", strip=True) if title else "",
                desc.get_text(separator=" ", strip=True) if desc else "",
                (
                    " ".join(
                        extra.get_text(separator=" ", strip=True)
                        for extra in extra_spans
                    )
                    if extra_spans
                    else ""
                ),
            )
        )
        courses_info.append(course_info)

    return courses_info


# IMPORTANT: only run this script once when initially setting up db, otherwise will duplicate entries
def main():
    base_url = "https://catalogs.northwestern.edu"
    main_url = f"{base_url}/undergraduate/courses-az/"
    links = get_links_from_sitemap(main_url)

    all_courses_info = []
    for relative_link in links:
        full_link = base_url + relative_link
        courses_info = extract_courseblock_text(full_link)
        all_courses_info.extend(courses_info)
    for title, description, extra_info in all_courses_info:
        # store course in chroma
        store_course_info(title, description, extra_info)


if __name__ == "__main__":
    main()
