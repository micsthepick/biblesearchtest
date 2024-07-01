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
BATCHSIZE = 64
testing_key = 'Password12344321'
AUTH = os.getenv("OPENAI_AI_KEY", testing_key)
testing_api = "http://127.0.0.1:8080"
api = os.getenv("OPENAI_API_ENDPOINT", testing_api)
route = "completion"
URL = f"{api}/{route}"
yes_token = " yes"
no_token = " no"
tokroute = 'tokenize'
TOKURL = f"{api}/{tokroute}"

# File paths
file_path = './Bible-kjv/Books.json'

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}


async def get_tok(session, tok):
    data = {"content": tok}
    async with session.post(TOKURL, headers=headers, json=data, ssl=False) as response:
        return (await response.json()).get("tokens", [-1])[-1]

async def load_books():
    async with aiofiles.open(file_path, 'r') as file:
        return json.loads(await file.read())

async def get_verses(verses_object):
    """ Generator to yield verse number and text from a chapter. """
    for verse in verses_object:
        vers = verse.get("verse", "???")
        text = verse.get("text", "Verse text missing!?")
        yield int(vers), text

async def get_chapters(book_object):
    """ Generator to yield chapter number and verses from a book. """
    for chapter_object in book_object.get("chapters", {}):
        chapt = chapter_object.get("chapter", "???")
        verses_object = chapter_object.get("verses", [])
        yield int(chapt), get_verses(verses_object)

async def get_books(books=None, path="Bible-kjv"):
    """ Generator to yield book name and its chapters. """
    if not books:
        books = ALL_BOOKS
    for book in tqdm(books, 'books', leave=False):
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
        yield book, get_chapters(book_object)

async def get_data(question, hunk):
    return f"""[INST]Determine whether the Bible text is applicable for answering the provided question[/INST]
<Question>{question}</Question>
<BibleText>{hunk}</BibleText>
(Your Answer Must be 'yes' or 'no' without quotes):"""

def get_score(value):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value) / log(1001)))}/1000"

async def generate_tasks(question, book_filter):
    async for book, book_contents in get_books(book_filter):
        hunk = ""
        hunk_start = (1, 1)
        async for chapter, chapter_contents in book_contents:
            async for verse, verse_text in chapter_contents:
                hunk += verse_text + '\n'
                if len(hunk) > HUNKSIZE:
                    yield (question, hunk, book, hunk_start, (chapter, verse))
                    hunk = ""
                    hunk_start = (chapter, verse + 1)
        if hunk:
            yield (question, hunk, book, hunk_start, (chapter, verse))

async def process_batch(session, values, yes_token_id, no_token_id):
    """ Send request to the API and get the response. """
    data = {
        "prompt": [await get_data(*value[:2]) for value in values],
        "temperature": -1,
        "n_predict": 1,
        "logit_bias": [[i,False] for i in range(256*256) if i not in [yes_token_id, no_token_id]],
        "n_probs": 2,
        "add_bos_token": True,
        "samplers": []
    }

    async with session.post(URL, headers=headers, json=data, ssl=False) as response:
        response_json = await response.json()

        if not isinstance(response_json, dict) or 'err' in response_json or 'error' in response_json:
            tqdm.write(str(response_json))
            return {
                "score": -999,
                "error": response_json
            }

    retvals = []
    for response, (question, contents_string, book, (chapter_start, verse_start), (chapter_end, verse_end)) in zip(response_json.get('results', []), values):
        resp_completions = response.get("completion_probabilities", [{}])[0].get("probs", [])
        if not resp_completions:
            return {
                "score": -999,
                "error": "completion_probabilities not found"
            }

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

        if  (yes_raw + no_raw) == 0:
            score = 0
        else:
            score = yes_raw / (yes_raw + no_raw)

        contents_string = f"{book} {chapter_start}:{verse_start}"
        if chapter_start != chapter_end:
            contents_string += f"-{chapter_end}:{verse_end}"
        elif verse_start != verse_end:
            contents_string += f"-{verse_end}"

        retvals.append({
            "score": score,
            "question": question,
            "book": book,
            "ref": contents_string,
            "chapter_start": chapter_start,
            "verse_start": verse_start,
            "chapter_end": chapter_end,
            "verse_end": verse_end
        })
    return retvals

async def main():
    book_filter = None
    if len(sys.argv) > 1:
        book_filter = sys.argv[1:]

    yes_token_id = None
    no_token_id = None
    while True:
        question = input('Search Query (e.g. question or biblical statement): ')

        async with aiohttp.ClientSession() as session:
            if (yes_token_id is None):
                yes_token_id = await get_tok(session, yes_token)
            if (no_token_id is None):
                no_token_id = await get_tok(session, no_token)
            scores = []
            task_gen = generate_tasks(question, book_filter)
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
                scores += await process_batch(session, tasks, yes_token_id, no_token_id)
            n = 7
            tqdm.write(f'Scores accumulated. Best {n} hunks to follow')
            best = sorted(scores, key=lambda x:-x['score'])[:n]
            tqdm.write('\n'.join([f"{get_score(obj['score'])}: {obj['ref']}" for obj in best]))
            for selection in best:
                tqdm.write(f"Selecting hunk: {selection['ref']}")
                specific_scores = []
                texts = []
                batch = []
                batchi = 0
                async for book, book_contents in get_books([selection['book']]):
                    async for chapter, chapter_contents in book_contents:
                        if chapter < selection['chapter_start']:
                            continue
                        elif chapter > selection['chapter_end']:
                            break
                        async for verse, verse_text in chapter_contents:
                            if chapter == selection['chapter_start'] and verse < selection['verse_start']:
                                continue
                            elif chapter == selection['chapter_end'] and verse > selection['verse_end']:
                                break
                            texts.append(verse_text)
                            batch.append((question, verse_text, book, (chapter, verse), (chapter, verse)))
                            batchi += 1
                            if batchi >= BATCHSIZE:
                                specific_scores += await process_batch(session, batch)
                                batch = []
                                batchi = 0
                if batch:
                        specific_scores += await process_batch(session, batch, yes_token_id, no_token_id)
                        batch = []
                nv = 3
                top_indexes = sorted(range(len(specific_scores)), key=lambda x: specific_scores[x]['score'], reverse=True)[:nv]
                tqdm.write(f"Best {nv} verses from hunk in {selection['book']}:")
                for i in top_indexes:
                    text = texts[i]
                    obj = specific_scores[i]
                    score = get_score(obj['score'])
                    ref = obj['ref']
                    tqdm.write(f'  Score: {score}, Reference: {ref};')
                    tqdm.write('    ' + '    '.join(text.split('\n')))


try:
    ALL_BOOKS = asyncio.run(load_books())
except FileNotFoundError:
    tqdm.write(f"Error: The file {file_path} does not exist.")
    sys.exit(1)
except json.JSONDecodeError:
    tqdm.write(f"Error: The file {file_path} is not a valid JSON file.")
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
