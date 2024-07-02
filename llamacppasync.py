import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm
import os
import json
import sys
from pathlib import Path
from math import log
import heapq


# Configuration and Constants
HUNKSIZE = 3696
BATCHSIZE = 32
# used model ctx size should be related to the above with the following eqn:
# CTXSIZE = BATCHSIZE*(HUNKSIZE/4+400/4), or alternatively HUNKSIZE = 4*CTXSIZE/BATCHSIZE-400
# (BATCHSIZE = 32, CTXSIZE = 32768 (max), HUNKSIZE = 3696) with HelloBible works well on my RTX 3090 with 24GB VRAM

testing_key = 'Password12344321'
AUTH = os.getenv("OPENAI_AI_KEY", testing_key)
testing_api = "http://127.0.0.1:8080"
api = os.getenv("OPENAI_API_ENDPOINT", testing_api)
route = "completion"
URL = f"{api}/{route}"
yes_token = "yes"
no_token = "no"
tokroute = 'tokenize'
TOKURL = f"{api}/{tokroute}"

# File paths
file_path = './Bible-kjv/Books.json'

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}
def get_data(question, hunk):
    return f"""[INST]Determine whether the Bible text is applicable for answering the provided question:
[TEXT]
{hunk}
[END TEXT]
(Your Answer Must be 'yes' or 'no' without quotes)
[QUESTION]
{question}
[END QUESTION]
[/INST]
Answer:"""

async def get_tok(session, tok):
    data = {"content": tok}
    async with session.post(TOKURL, headers=headers, json=data, ssl=False) as response:
        return (await response.json()).get("tokens", [-1])[-1]

async def load_books():
    async with aiofiles.open(file_path, 'r') as file:
        return json.loads(await file.read())

def get_verses(verses_object):
    """ Generator to yield verse number and text from a chapter. """
    for verse in verses_object:
        vers = verse.get("verse", "???")
        text = verse.get("text", "Verse text missing!?")
        yield int(vers), text

def get_chapters(book_object):
    """ Generator to yield chapter number and verses from a book. """
    for chapter_object in book_object.get("chapters", []):
        chapt = chapter_object.get("chapter", "???")
        verses_object = chapter_object.get("verses", [])
        yield int(chapt), get_verses(verses_object)

async def get_books(books=None, path="Bible-kjv"):
    """ Generator to yield book name and its chapters. """
    if not books:
        books = ALL_BOOKS
    for book in books:
        file_path = Path(path).joinpath(f"{book.replace(' ', '')}.json")
        try:
            async with aiofiles.open(file_path, 'r') as file:
                book_object = json.loads(await file.read())
        except FileNotFoundError:
            tqdm.write(f"Error: The file {file_path} does not exist.")
            continue
        except json.JSONDecodeError:
            tqdm.write(f"Error: The file {file_path} is not a valid JSON file.")
            continue
        yield book, get_chapters(book_object)

def get_score(value):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value) / log(1001)))}/1000"

async def generate_tasks(queue, book_filter):
    book_count = len(book_filter if book_filter else ALL_BOOKS)
    async for book, book_contents in tqdm(get_books(book_filter), desc="Books: ", total=book_count, leave=True):
        hunk = ""
        hunk_start_chapter = 1
        hunk_start_verse = 1
        async for chapter, chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
            async for verse, verse_text in tqdm(list(chapter_contents), desc="Verses: ", leave=False):
                hunk += verse_text + '\n'
                if len(hunk) > HUNKSIZE:
                    await queue.put((hunk, book, hunk_start_chapter, hunk_start_verse, chapter, verse))
                    hunk = ""
                    hunk_start_chapter = chapter
                    hunk_start_verse = verse + 1
        if hunk:
            await queue.put((hunk, book, hunk_start_chapter, hunk_start_verse, chapter, verse))

    tqdm.write('final tasks will finish shortly')
    await queue.put(None)  # Signal the end of the queue

async def process(queue, session, question, yes_token_id, no_token_id, topn=25):
    """ Process items from the queue and send requests to the API. """
    results = []

    def append_if_good(results, elem, topn=topn):
        return heapq.nlargest(topn, results + [elem], key=lambda x:x['score'])

    while True:
        item = await queue.get()
        if item is None:
            break

        hunk, book, chapter_start, verse_start, chapter_end, verse_end = item

        data = {
            "prompt": get_data(question, hunk),
            "temperature": -1,
            "n_predict": 1,
            "logit_bias": [[i, False] for i in range(256 * 256) if i not in [yes_token_id, no_token_id]],
            "n_probs": 2,
            "add_bos_token": True,
            "samplers": []
        }
        async with session.post(URL, headers=headers, json=data, ssl=False) as response:
            response_json = await response.json()

        if not isinstance(response_json, dict) or 'err' in response_json or 'error' in response_json:
            tqdm.write(str(response_json))
            continue

        resp_completions = response_json.get("completion_probabilities", [{}])[0].get("probs", None)
        if not resp_completions:
            tqdm.write("ERR: no completions")
            continue

        yes_raw = 0
        no_raw = 0

        for tok in resp_completions:
            if not isinstance(tok, dict):
                break
            if tok.get("tok_str", "").strip() == yes_token.strip():
                yes_raw = tok.get('prob', 0)
            elif tok.get("tok_str", "").strip() == no_token.strip():
                no_raw = tok.get('prob', 0)

        if yes_raw is None:
            yes_raw = 0
        if no_raw is None:
            no_raw = 0

        if (yes_raw + no_raw) == 0:
            score = 0
        else:
            score = yes_raw / (yes_raw + no_raw)

        contents_string = f"{book} {chapter_start}:{verse_start}"
        if chapter_start != chapter_end:
            contents_string += f"-{chapter_end}:{verse_end}"
        elif verse_start != verse_end:
            contents_string += f"-{verse_end}"

        results = append_if_good(
            results,
            {
                "score": score,
                "question": question,
                "book": book,
                "ref": contents_string,
                "verse": None if chapter_end != chapter_start or verse_end != verse_start else hunk,
                "chapter_start": chapter_start,
                "verse_start": verse_start,
                "chapter_end": chapter_end,
                "verse_end": verse_end
            }
        )

    return results

async def get_tasks_for_selection(queue, selection):
    async for book, book_contents in get_books([selection['book']]):
        async for chapter, chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
            if chapter < selection['chapter_start']:
                continue
            elif chapter > selection['chapter_end']:
                break
            async for verse, verse_text in tqdm(list(chapter_contents), desc="Verses: ", leave=False):
                if chapter == selection['chapter_start'] and verse < selection['verse_start']:
                    continue
                elif chapter == selection['chapter_end'] and verse > selection['verse_end']:
                    break
                await queue.put((verse_text, book, chapter, verse, chapter, verse))
    await queue.put(None)

async def main():
    queue = asyncio.Queue(BATCHSIZE)

    book_filter = None
    if len(sys.argv) > 1:
        book_filter = sys.argv[1:]

    yes_token_id = None
    no_token_id = None

    while True:
        question = input('Search Query (e.g. question or biblical statement): ')

        base_len = len(get_data(question, ""))

        if base_len > 400:
            if input(f'{base_len-400} chars over limit! continue anyways? [Y/n]')[0].lower() == 'n':
                continue

        async with aiohttp.ClientSession() as session:
            if yes_token_id is None:
                yes_token_id = await get_tok(session, yes_token)
            if no_token_id is None:
                no_token_id = await get_tok(session, no_token)

            producer = generate_tasks(queue, book_filter)
            consumer = process(queue, session, question, yes_token_id, no_token_id, 5)
            scores = (await asyncio.gather(*[producer, consumer]))[1]
            
            print(f'Scores accumulated. Best {len(scores)} hunks to follow')
            # already sorted by heapq
            print('\n'.join([f"{get_score(obj['score'])}: {obj['ref']}" for obj in scores]))
            for selection in scores:
                print(f"Selecting hunk: {selection['ref']}")
                nv = 3
                producer = get_tasks_for_selection(queue, selection)
                consumer = process(queue, session, question, yes_token_id, no_token_id, nv)
                scores = (await asyncio.gather(*[producer, consumer]))[1]
                print(f"Best {len(scores)} verses from hunk in {selection['book']}:")
                for obj in scores:
                    text = obj['verse']
                    score = get_score(obj['score'])
                    ref = obj['ref']
                    print(f'  Score: {score}, Reference: {ref};')
                    print('    ' + '    '.join(text.split('\n')))


try:
    ALL_BOOKS = asyncio.run(load_books())
except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: The file {file_path} is not a valid JSON file.")
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
