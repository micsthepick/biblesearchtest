import requests
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from math import log

# Configuration and Constants
HUNKSIZE = 4000
testing_key = 'Password12344321'
AUTH = os.getenv("OPENAI_AI_KEY", testing_key)
testing_api = "http://127.0.0.1:5000"
api = os.getenv("OPENAI_API_ENDPOINT", testing_api)
route = "v1/internal/logits"
URL = f"{api}/{route}"

# File paths
file_path = './Bible-kjv/Books.json'

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}

# Load the Books JSON file
try:
    with open(file_path, 'r') as file:
        ALL_BOOKS = json.load(file)
except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: The file {file_path} is not a valid JSON file.")
    sys.exit(1)


def get_verses(verses_object: dict, chname: str):
    """ Generator to yield verse number and text from a chapter. """
    for verse in tqdm(verses_object, desc=f"Verses of chapter {chname}", leave=False):
        vers = verse.get("verse", "???")
        text = verse.get("text", "Verse text missing!?")
        yield int(vers), text

def get_chapters(book_object: dict, bname: str):
    """ Generator to yield chapter number and verses from a book. """
    for chapter_object in tqdm(book_object.get("chapters", []), desc=f"Book {bname}", leave=False):
        chapt = chapter_object.get("chapter", "???")
        verses_object = chapter_object.get("verses", [])
        yield int(chapt), get_verses(verses_object, chapt)

def get_books(books: None | list[str]=None, path: str="Bible-kjv"):
    """ Generator to yield book name and its chapters. """
    if not books:
        books = ALL_BOOKS
    for book in tqdm(books, desc=f'Searching {len(books)} KJV books'):
        file_path = Path(path).joinpath(f"{book.replace(' ', '')}.json")
        try:
            with open(file_path, 'r') as file:
                book_object = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            continue
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is not a valid JSON file.")
            continue
        yield book, get_chapters(book_object, book)
    print('\n')

def do_request(question: str, hunk: str, book: str, hunk_start: tuple[int, int], hunk_end: tuple[int, int]):
    """ Send request to the API and get the response. """
    data = {
        "prompt": (
f"""[INST]Determine whether the Bible text is applicable for answering the provided question[/INST]
[QUESTION]{question}[/QUESTION]
[TEXT]{hunk}[/TEXT]
Answer (Must be 'yes' or 'no' without quotes):"""
        ),
        "custom_token_bans": ','.join(str(i) for i in range(256*256) if i not in [5081, 708]),
        "top_logits": 2,
        "add_bos_token": True,
        "use_samplers": True
    }

    try:
        # Send POST request
        response = requests.post(URL, headers=headers, json=data, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)

    # Extract probabilities for "yes" and "no"
    response_json = response.json()
    if not isinstance(response_json, dict):
        print(response_json)
        sys.exit(1)

    yes_raw = response_json.get('▁yes', 0)
    no_raw = response_json.get('▁no', 0)
    assert yes_raw + no_raw != 0

    score = yes_raw
    chapter_start, verse_start = hunk_start
    chapter_end, verse_end = hunk_end

    contents_string = f"{book} {chapter_start}:{verse_start}"
    if chapter_start != chapter_end:
        contents_string += f"-{chapter_end}:{verse_end}"
    elif verse_start != verse_end:
        contents_string += f"-{verse_end}"

    return {
        "score": score,
        "book": book,
        "ref": contents_string,
        "chapter_start": chapter_start,
        "verse_start": verse_start,
        "chapter_end": chapter_end,
        "verse_end": verse_end
    }

def get_score(value: float):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value) / log(1001)))}/1000"

def main():
    while True:
        scores = []
        question = input('Search Query (e.g. question or biblical statement): ')
        for book, book_contents in get_books():
            hunk = ""
            hunk_start = (1, 1)
            for chapter, chapter_contents in book_contents:
                for verse, verse_text in chapter_contents:
                    hunk += verse_text + '\n'
                    if len(hunk) > HUNKSIZE:
                        scores.append(do_request(question, hunk, book, hunk_start, (chapter, verse)))
                        hunk = ""
                        hunk_start = (chapter, verse + 1)
            # if we end up with a remaining hunk at the end of a book
            if hunk:
                scores.append(do_request(question, hunk, book, hunk_start, (chapter, verse)))
                hunk = ""
                hunk_start = (chapter, verse + 1)

        n = 5
        print(f'Scores accumulated. Best {n} hunks to follow')
        best = sorted(scores, key=lambda x:-x['score'])[:n]
        print(*(f"{get_score(obj['score'])}: {obj['ref']}" for obj in best), sep='\n')
        for selection in best:
            print(f'Selecting hunk:', selection['ref'])
            specific_scores = []
            for book, book_contents in get_books([selection['book']]):
                for chapter, chapter_contents in book_contents:
                    if chapter < selection['chapter_start']:
                        continue
                    elif chapter > selection['chapter_end']:
                        break
                    for verse, verse_text in chapter_contents:
                        if chapter == selection['chapter_start'] and verse < selection['verse_start']:
                            continue
                        elif chapter == selection['chapter_end'] and verse > selection['verse_end']:
                            break
                        specific_scores.append((do_request(question, verse_text, book, (chapter, verse), (chapter, verse)), verse_text))

        nv = 1
        print(f"Best {nv} verses from hunk in {selection['book']}:")
        for obj, text in sorted(specific_scores, key=lambda x: -x[0]['score'])[:nv]:
            score = get_score(obj['score'])
            ref = obj['ref']
            print(f'  Score: {score}, Reference: {ref};')
            print('    ' + '    '.join(text.split('\n')))


if __name__ == "__main__":
    main()