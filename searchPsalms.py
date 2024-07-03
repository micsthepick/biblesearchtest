import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm
import os
import json
import sys
from math import log
import heapq


# Configuration and Constants
BATCHSIZE = 32
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

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}
def get_data(question, hunk):
    return f"""[INST]Determine whether the Bible text is applicable for QUERY:[/INST]
[TEXT]
{hunk}
[/TEXT]
(Your Answer Must be 'yes' or 'no' without quotes)
[QUERY]
{question}
[/QUERY]
Answer:"""


async def get_tok(session, tok):
    data = {"content": tok}
    async with session.post(TOKURL, headers=headers, json=data, ssl=False) as response:
        return (await response.json()).get("tokens", [-1])[-1]

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

def get_score(value):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value) / log(1001)))}/1000"

async def generate_tasks(queue):
    async with aiofiles.open("Bible-kjv/Psalms.json", 'r') as file:
        book_contents = json.loads(await file.read())
    async for chapter, chapter_contents in tqdm(list(get_chapters(book_contents)), desc="Chapters: ", leave=True):
        hunk = ""
        async for _, verse_text in tqdm(list(chapter_contents), desc="Verse: ", leave=False):
            hunk += verse_text + '\n'
        if hunk:
            await queue.put((chapter, hunk))

    tqdm.write('final tasks will finish shortly')
    await queue.put(None)  # Signal the end of the queue

async def process(queue, session, question, yes_token_id, no_token_id, topn=25):
    """ Process items from the queue and send requests to the API. """
    results = []

    def append_if_good(results, elem, topn=topn):
        return heapq.nlargest(topn, results + [elem], key=lambda x:x['score'])

    while True:
        stuff = await queue.get()

        if stuff is None:
            break

        chapter, hunk = stuff

        data = {
            "prompt": get_data(question, chapter),
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

        contents_string = f"Psalm {chapter}"

        results = append_if_good(
            results,
            {
                "score": score,
                "question": question,
                "book": "Psalms",
                "ref": contents_string,
                "chapter": hunk,
            }
        )

    return results

async def main():
    queue = asyncio.Queue(BATCHSIZE)

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

            producer = generate_tasks(queue)
            consumer = process(queue, session, question, yes_token_id, no_token_id, 5)
            scores = (await asyncio.gather(*[producer, consumer]))[1]

            print(f'Scores accumulated. Best {len(scores)} psalms to follow')
            for psalm in scores:
                print(f"{psalm['ref']} is score {get_score(psalm['score'])}:")
                print(psalm['chapter'])
                print('\n')

if __name__ == "__main__":
    asyncio.run(main())
