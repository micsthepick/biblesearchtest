import asyncio
import aiohttp
import aiofiles
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from math import log

# Configuration and Constants
HUNKSIZE = 4000
BATCHSIZE = 4
testing_key = 'Password12344321'
AUTH = os.getenv("OPENAI_AI_KEY", testing_key)
testing_api = "http://127.0.0.1:8000"
api = os.getenv("OPENAI_API_ENDPOINT", testing_api)
route = "v1/chat/completions"
URL = f"{api}/{route}"
yes_token = "▁yes"
yes_token_id = 4874
no_token = "▁no"
no_token_id = 694

# File paths
file_path = './Bible-kjv/Books.json'

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}

async def load_books():
    async with aiofiles.open(file_path, 'r') as file:
        return json.loads(await file.read())

try:
    ALL_BOOKS = asyncio.run(load_books())
except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: The file {file_path} is not a valid JSON file.")
    sys.exit(1)

async def get_verses(verses_object, chname):
    """ Generator to yield verse number and text from a chapter. """
    for verse in tqdm(verses_object, f'verses of ch {chname}'):
        vers = verse.get("verse", "???")
        text = verse.get("text", "Verse text missing!?")
        yield int(vers), text

async def get_chapters(book_object, bname):
    """ Generator to yield chapter number and verses from a book. """
    for chapter_object in tqdm(book_object.get("chapters", []), f'capters of {bname}'):
        chapt = chapter_object.get("chapter", "???")
        verses_object = chapter_object.get("verses", [])
        yield int(chapt), get_verses(verses_object, chapt)

async def get_books(books=None, path="Bible-kjv"):
    """ Generator to yield book name and its chapters. """
    if not books:
        books = ALL_BOOKS
    for book in tqdm(books, 'books'):
        file_path = Path(path).joinpath(f"{book.replace(' ', '')}.json")
        try:
            async with aiofiles.open(file_path, 'r') as file:
                book_object = json.loads(await file.read())
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
            continue
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is not a valid JSON file.")
            continue
        yield book, get_chapters(book_object, book)
    print('\n')

async def do_request(session, question, hunk, book, hunk_start, hunk_end):
    """ Send request to the API and get the response. """
    data = {
        "messages": [
            {"role": "system", "content":
f"""Determine whether the Bible text is applicable for answering the provided question
[QUESTION]{question}[/QUESTION]
[TEXT]{hunk}[/TEXT]
Answer must be 'yes' or 'no' without quotes:"""
            }
        ],
        "logits": {str(i):-100 for i in range(256*256) if i not in [yes_token_id, no_token_id]},
        "top_logits": 2,
        "add_bos_token": True,
        "use_samplers": True
    }

    chapter_start, verse_start = hunk_start
    chapter_end, verse_end = hunk_end

    contents_string = f"{book} {chapter_start}:{verse_start}"
    if chapter_start != chapter_end:
        contents_string += f"-{chapter_end}:{verse_end}"
    elif verse_start != verse_end:
        contents_string += f"-{verse_end}"

    async with session.post(URL, headers=headers, json=data, ssl=False) as response:
        response_json = await response.json()
        if not isinstance(response_json, dict):
            print(response_json)
            return {
                "score": -999,
                "book": book,
                "ref": contents_string,
                "chapter_start": chapter_start,
                "verse_start": verse_start,
                "chapter_end": chapter_end,
                "verse_end": verse_end,
                "error": response_json
            }

        yes_raw = response_json.get(yes_token, 0)
        no_raw = response_json.get(no_token, 0)
        assert yes_raw + no_raw != 0

        score = yes_raw

        return {
            "score": score,
            "book": book,
            "ref": contents_string,
            "chapter_start": chapter_start,
            "verse_start": verse_start,
            "chapter_end": chapter_end,
            "verse_end": verse_end
        }

def get_score(value):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value) / log(1001)))}/1000"

async def generate_tasks(session, question):
    async for book, book_contents in get_books():
        hunk = ""
        hunk_start = (1, 1)
        async for chapter, chapter_contents in book_contents:
            async for verse, verse_text in chapter_contents:
                hunk += verse_text + '\n'
                if len(hunk) > HUNKSIZE:
                    yield do_request(session, question, hunk, book, hunk_start, (chapter, verse))
                    hunk = ""
                    hunk_start = (chapter, verse + 1)
        if hunk:
            yield do_request(session, question, hunk, book, hunk_start, (chapter, verse))

async def main():
    question = input('Search Query (e.g. question or biblical statement): ')

    async with aiohttp.ClientSession() as session:
        scores = []
        task_gen = generate_tasks(session, question)
        iterating = True
        while iterating:
            tasks = []
            for _ in range(BATCHSIZE):
                try:
                    val = await task_gen.__anext__()
                    tasks.append(val)
                except StopAsyncIteration:
                    iterating = False
                    break
            scores += await asyncio.gather(*tasks)
    print(scores[:2])
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
                    specific_scores.append((await do_request(question, verse_text, book, (chapter, verse), (chapter, verse)), verse_text))

    nv = 1
    print(f"Best {nv} verses from hunk in {selection['book']}:")
    for obj, text in sorted(specific_scores, key=lambda x: -x[0]['score'])[:nv]:
        score = get_score(obj['score'])
        ref = obj['ref']
        print(f'  Score: {score}, Reference: {ref};')
        print('    ' + '    '.join(text.split('\n')))

if __name__ == "__main__":
    asyncio.run(main())
