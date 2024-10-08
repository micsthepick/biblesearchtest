import asyncio
import zipfile
import aiohttp
from math import ceil
import aiofiles
from datetime import datetime, timedelta
import discord
from discord import Intents
from discord.ext import commands
from discord.ui import View
from tqdm.asyncio import tqdm
import os
import json
import sys
from pathlib import Path
from math import log
import heapq
from bs4 import BeautifulSoup


# Configuration and Constants
HUNKSIZE = 11888
BATCHSIZE = 32
# used model interaction size should be related to
# the above with the following eqn:
# CTXSIZE = BATCHSIZE*(HUNKSIZE/4+400/4),
#   or alternatively HUNKSIZE = 3*CTXSIZE/BATCHSIZE-400
# (BATCHSIZE = 8, CTXSIZE = 32768 (max), HUNKSIZE = 15984
#   with HelloBiblev0.2 works well on my RTX 3090 with 24GB VRAM)

testing_key = 'Password12344321'
AUTH = os.getenv("OPENAI_AI_KEY", testing_key)
testing_api = "http://127.0.0.1:8080"
api = os.getenv("OPENAI_API_ENDPOINT", testing_api)
route = "completion"
URL = f"{api}/{route}"
yes_tokens = [" yes", "yes", " Yes", "Yes"]
no_tokens = [" no", "no", " No", "No"]
tokroute = 'tokenize'
TOKURL = f"{api}/{tokroute}"
bot_token = os.getenv("DISCORD_KEY", None)
channel_name = "bible-search"

api = os.getenv("OPENAI_API_ENDPOINT", testing_api)


class LockWithFuture(asyncio.Lock):
    def __init__(self):
        self.future = None
        super().__init__()

    def add_future(self):
        """should be called after aquiring the lock"""
        if self.future is None:
            self.future = asyncio.Future()
        else:
            raise ValueError('Setting future of LockWithFuture twice!')

    async def try_get_future(self):
        try:
            return await self.future
        except AttributeError:
            return

    def try_resolve_future(self, value=None):
        try:
            self.future.set_result(value)
            self.future = None
        except AttributeError:
            pass


class TimeoutLock:
    def __init__(self, duration=120):
        self.release_time = None
        self.duration = duration
        self._lock = asyncio.Lock()
        self._waiting = False

    async def try_acquire(self, ):
        async with self._lock:
            if self._waiting:
                return False
            self._waiting = True
            return True

    async def finished(self):
        async with self._lock:
            self._waiting = False

    async def can_start_in(self, newduration=None):
        async with self._lock:
            if newduration:
                self.duration = newduration
            if self.release_time is None or self.release_time <= datetime.now():
                return None
            return self.release_time - datetime.now()

    async def release(self):
        async with self._lock:
            self.release_time = None

    async def start(self):
        async with self._lock:
            self.release_time = datetime.now() + timedelta(seconds=self.duration)


# Twitch bot setupF
bot = commands.Bot(
    intents=Intents(messages=True),
    command_prefix='!',
)

# Headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}"
}


def get_data(question, hunk):
    return f"""System: As far as possible, refer to the stories in the bible. Determine whether the provided text is applicable for answering the provided query.
[BEGIN TEXT]
{hunk}
[END TEXT]

User: Does this bible text help to find answers to {question}

Assistant:"""


async def get_tok(session, toks):
    tokIds = set()
    for tok in toks:
        data = {"content": tok}
        async with session.post(TOKURL, headers=headers, json=data, ssl=False) as response:
            toks = (await response.json()).get("tokens", [-1])
            print(f'"{tok}" tokens: {toks}')
            tokIds.add(toks[0])
    return tokIds


async def load_books(file_path):
    async with aiofiles.open(file_path, 'r') as file:
        return json.loads(await file.read())


def get_bible_verses(verses_object):
    """ Generator to yield verse number and text from a chapter. """
    for verse in verses_object:
        vers = verse.get("verse", "???")
        text = verse.get("text", "Verse text missing!?")
        yield vers, text


def get_bible_chapters(book_object):
    """ Generator to yield chapter number and verses from a book. """
    for chapter_object in book_object.get("chapters", []):
        chapt = chapter_object.get("chapter", "???")
        verses_object = chapter_object.get("verses", [])
        yield chapt, get_bible_verses(verses_object)


def get_bible_books(books=None, path="Bible-kjv"):
    """ Generator to yield book name and its chapters. """
    if not books:
        books = [{'book_id': book} for book in KJV_BOOK_DETAILS]
    for book in books:
        file_path = Path(path).joinpath(
            f"{book['book_id'].replace(' ', '')}.json")
        try:
            with open(file_path, 'r') as file:
                book_object = json.loads(file.read())
        except FileNotFoundError:
            tqdm.write(f"Error: The file {file_path} does not exist.")
            continue
        except json.JSONDecodeError:
            tqdm.write(
                f"Error: The file {file_path} is not a valid JSON file.")
            continue
        yield book, get_bible_chapters(book_object)


def get_egw_paragraphs(paras_object, from_pid, to_pid=None):
    """ Generator to yield verse number and text from a chapter. """
    started = False
    for para in paras_object:
        para_id = para.get("para_id", "???")
        if para_id == from_pid:
            started = True
        elif not started:
            continue
        if para_id == to_pid:
            break
        text = para.get("content")
        refcode_2 = para.get('refcode_2', '')
        refcode_3 = para.get('refcode_3', '')
        refcode_4 = para.get('refcode_4', False)
        if refcode_4:
            continue
        yield (refcode_2, refcode_3), text


def get_egw_chapters(zip_ref: zipfile.ZipFile, index):
    """ Generator to yield chapter number and verses from a book."""
    for chapter_object in index:
        para_id = chapter_object.get("para_id")
        id_next = chapter_object.get("id_next", None)
        fname = f"{para_id}.json"
        if fname in set(zi.filename for zi in zip_ref.infolist()):
            with zip_ref.open(fname, 'r') as para_file:
                paras_object = json.load(para_file)
        yield get_egw_paragraphs(paras_object, from_pid=para_id, to_pid=id_next)


def get_egw_books(books=None, path="egwbooks"):
    """Async generator to yield book name and its chapters."""
    if not books:
        books = EGW_BOOK_DETAILS
    for book in books:
        file_path = Path(path).joinpath(f"{book['book_id']}.egwbook")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                with zip_ref.open('toc.json', 'r') as index_file:
                    index = json.load(index_file)
                yield book, get_egw_chapters(zip_ref, index)
        except FileNotFoundError:
            tqdm.write(f"Error: The file {file_path} does not exist.")
            continue


def get_score(value):
    """ Convert raw score to a human-readable score. """
    return f"{int(1000-round(1000*log(1001-1000*value['score']) / log(1001)))}/1000"


async def bible_gen_cb(book_filter):
    book_count = len(book_filter if book_filter else KJV_BOOK_DETAILS)
    for book, book_contents in tqdm(get_bible_books(book_filter), desc="Books: ", total=book_count, leave=False):
        hunk = ""
        hunk_start_chapter = None
        hunk_start_verse = None
        for chapter, chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
            for verse, verse_text in tqdm(list(chapter_contents), desc="Verses: ", leave=False):
                if hunk_start_verse is None:
                    hunk_start_verse = verse
                    hunk_start_chapter = chapter
                hunk += verse_text + '\n'
                if len(hunk) > HUNKSIZE:
                    yield (hunk, book, hunk_start_chapter, hunk_start_verse, chapter, verse)
                    hunk = ""
                    hunk_start_chapter = None
                    hunk_start_verse = None
        if hunk:
            yield (hunk, book, hunk_start_chapter, hunk_start_verse, chapter, verse)
            hunk = ""
            hunk_start_chapter = None
            hunk_start_verse = None


async def egw_gen_cb(book_filter):
    book_count = len(book_filter if book_filter else EGW_BOOK_DETAILS)
    for book, book_contents in tqdm(get_egw_books(book_filter), desc="Books: ", total=book_count, leave=False):
        hunk = ""
        hunk_start_chapter = None
        hunk_start_para = None
        for chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
            for (chapter, para), para_text in tqdm(list(chapter_contents), desc="Paragraphs: ", leave=False):
                if hunk_start_para is None:
                    hunk_start_para = para
                    hunk_start_chapter = chapter
                if chapter != hunk_start_chapter:
                    if hunk:
                        yield (hunk, book, hunk_start_chapter, hunk_start_para, chapter, para)
                    hunk = ""
                    hunk_start_chapter = None
                    hunk_start_para = None
                hunk += para_text + '\n'
                if len(hunk) > HUNKSIZE:
                    yield (hunk, book, hunk_start_chapter, hunk_start_para, chapter, para)
                    hunk = ""
                    hunk_start_chapter = None
                    hunk_start_para = None
        if hunk:
            yield (hunk, book, hunk_start_chapter, hunk_start_para, chapter, para)
            hunk = ""
            hunk_start_chapter = None
            hunk_start_para = None


async def process_and_add_to_scores(pbar: tqdm, pbarlock: LockWithFuture, results, item, session, question, book_sep, yes_token_ids, no_token_ids, topn):
    def append_if_good(results, elem):
        results[:] = heapq.nlargest(
            topn, results + [elem], key=lambda x: x['score'])

    hunk, book, chapter_start, verse_start, chapter_end, verse_end = item

    data = {
        "prompt": get_data(question, hunk),
        "temperature": -1,
        "n_predict": 1,
        "logit_bias": [[i, False] for i in range(256 * 256) if i not in yes_token_ids | no_token_ids],
        "n_probs": 20,
        "add_bos_token": True,
        "samplers": []
    }
    async with session.post(URL, headers=headers, json=data, ssl=False) as response:
        del data["logit_bias"]
        print(response, data)
        response_json = await response.json()

    if not isinstance(response_json, dict) or 'err' in response_json or 'error' in response_json:
        tqdm.write(str(response_json))
        return

    resp_completions = response_json.get(
        "completion_probabilities", [{}])[0].get("probs", None)

    if not resp_completions:
        tqdm.write("ERR: no completions")
        return

    yes_raw = 0
    no_raw = 0

    for tok in resp_completions:
        if not isinstance(tok, dict):
            break
        if tok.get("tok_str") in yes_tokens:
            yes_raw += tok.get('prob')
        elif tok.get("tok_str") in no_tokens:
            no_raw += tok.get('prob')

    if (yes_raw + no_raw) == 0:
        score = 0
    else:
        score = yes_raw / (yes_raw + no_raw)

    contents_string = f"{book['title']} {chapter_start}{book_sep}{verse_start}"
    if chapter_start != chapter_end:
        contents_string += f"-{chapter_end}{book_sep}{verse_end}"
    elif verse_start != verse_end:
        contents_string += f"-{verse_end}"

    async with pbarlock:
        pbar.n += 1
        pbar.refresh()
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
        pbarlock.try_resolve_future()


async def process(gen_obj, pbar, session, question, book_sep, yes_token_ids, no_token_ids, num):
    """ Process items from the generator and send requests to the API. """
    results = []

    pbarlock = LockWithFuture()
    tasks = set()
    exceptions = []

    # Use task.add_done_callback() to handle exceptions
    def task_done_callback(task):
        if task.exception():
            exceptions.append(task.exception())
        tasks.discard(task)
        del task

    try:
        async for item in gen_obj:
            waiting = False
            async with pbarlock:
                pbar.n -= 1
                pbar.refresh()
                if pbar.n == 0:
                    pbarlock.add_future()
                    waiting = True
            if waiting:
                await pbarlock.try_get_future()
            task = asyncio.create_task(process_and_add_to_scores(
                pbar, pbarlock, results, item, session,
                question, book_sep, yes_token_ids, no_token_ids, num))
            tasks.add(task)
            task.add_done_callback(task_done_callback)
    finally:
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        if exceptions:
            raise list(exceptions)[0]

    return results


async def send_safe(interaction: discord.Interaction, message, limit=1900):
    words = message.split(' ')
    next_message = []
    for word in words:
        if len(word) > limit:
            await interaction.followup.send(' '.join(next_message))
            n = 1 + (len(word) - 1) // limit
            next_words = [word[i:i + limit] for i in range(0, len(word), n)]
            for word in next_words[:-1]:
                await interaction.followup.send(word)
            next_message = next_words[-1]
        elif len(' '.join(next_message + [word])) > limit:
            await interaction.followup.send(' '.join(next_message))
            next_message = [word]
        next_message.append(word)
    if next_message:
        await interaction.followup.send(' '.join(next_message))


async def get_tasks_for_selection(generate_cb, selection):
    if generate_cb is egw_gen_cb:
        for book, book_contents in get_egw_books([selection['book']]):
            recording = False
            stopping = False
            for chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
                for (chapter, para), paragraph_contents in tqdm(list(chapter_contents), desc="Paragraphs: ", leave=False):
                    if para == selection['verse_start'] and chapter == selection['chapter_start']:
                        recording = True
                    if recording:
                        yield (paragraph_contents, book, chapter, para, chapter, para)
                    if para == selection['verse_end'] and chapter == selection['chapter_end']:
                        stopping = True
                        break
                if stopping:
                    break
    elif generate_cb is bible_gen_cb:
        for book, book_contents in get_bible_books([selection['book']]):
            for chapter, chapter_contents in tqdm(list(book_contents), desc="Chapters: ", leave=False):
                if chapter < selection['chapter_start']:
                    continue
                elif chapter > selection['chapter_end']:
                    break
                for verse, verse_text in tqdm(list(chapter_contents), desc="Verses: ", leave=False):
                    if chapter == selection['chapter_start'] and verse < selection['verse_start']:
                        continue
                    elif chapter == selection['chapter_end'] and verse > selection['verse_end']:
                        break
                    yield (verse_text, book, chapter, verse, chapter, verse)
    else:
        raise ValueError("bad callback")

yes_token_ids = set()
no_token_ids = set()

processingSem = asyncio.BoundedSemaphore()
timeoutLock = TimeoutLock()

TIMEOUT = 45


def normalize_string(str):
    replacements = {v: None for v in '\'"[]<>{}\u2018\u2019\u201c\u201d'} | {
        v: '-' for v in '_\u2014\u2013'} | {'\u00e9': 'e', '\u00ed': 'i'} | {'\u00a0': ' '}
    return ' '.join(str.translate(replacements).lower().split()).strip()


class CandidateButton(discord.ui.Button):
    async def callback(self, interaction: discord.Interaction):
        button_id = self.custom_id

        if button_id.startswith("select_"):
            index = int(button_id.split("_")[1])
            selected_candidate = self.view.candidate_list[index]
            self.view.selection.set_result(selected_candidate[0]['book_id'])
            self.view.clear_items()
            await self.view.interaction.edit_original_response(view=None)
            self.view.stop()
        elif button_id == 'next':
            self.view.page += 1
            await self.view.update_message()
        elif button_id == 'previous':
            self.view.page -= 1
            await self.view.update_message()
        else:
            raise ValueError("unkown button!")
        await interaction.response.defer()


class CandidateSelectView(View):
    def __init__(self, interaction, candidate_list):
        super().__init__(timeout=120)
        self.interaction = interaction
        self.candidate_list = heapq.nlargest(80, candidate_list, key=lambda x: x[1])
        self.per_page = 8
        self.page = 0
        self.pages = 1 + (len(self.candidate_list) - 1) // self.per_page
        self.selection = asyncio.Future()

    def on_timeout(self):
        self.clear_items()
        self.selection.set_result(None)

    def update_buttons(self):
        self.clear_items()

        start = self.page * self.per_page
        end = start + self.per_page
        candidates = self.candidate_list[start:end]

        self.add_item(
            CandidateButton(
                label="Previous", row=4,
                style=discord.ButtonStyle.primary,
                custom_id="previous",
                disabled=(self.page <= 0)
            )
        )
        self.add_item(
            CandidateButton(
                label="Next",
                row=4,
                style=discord.ButtonStyle.primary,
                custom_id="next",
                disabled=(end >= len(self.candidate_list))
            )
        )

        for idx, candidate in enumerate(candidates):
            title = candidate[0]['title']
            if len(title) > 75:
                title = title[:75] + '...'
            self.add_item(CandidateButton(label=title, row=idx // 2, style=discord.ButtonStyle.secondary, custom_id=f"select_{start + idx}"))

    async def update_message(self):

        content = f"Choose the best candidate: (page {self.page+1}/{self.pages})\n"

        self.update_buttons()

        await self.interaction.edit_original_response(content=content, view=self)


async def choose_candidates(interaction, candidate_list):
    if not candidate_list:
        return
    view = CandidateSelectView(interaction, candidate_list)
    await view.update_message()
    selection = await view.selection
    return selection


def geteid():
    diff = (datetime.now() - datetime(2020, 1, 1))
    denom = 131071
    base = (int(diff.total_seconds()) ^ diff.microseconds) % denom
    return (((base << 31) % denom) - (base % denom)) % denom


async def do_error(interaction: discord.Interaction, e: Exception):
    if hasattr(e, 'has_been_returned_via_discord') and e.has_been_returned_via_discord:
        return
    e.has_been_returned_via_discord = True
    print(e)
    excid = geteid()
    await interaction.followup.send(content=f'An error occurred, please give the following ID to the bot owner: {excid}.')
    print(f'>>>EXEC {excid}<<<')


async def do_search(interaction: discord.Interaction, generate_cb, book_sep, user_name, query, details):
    global yes_token_ids, no_token_ids
    send_cb = interaction.edit_original_response
    try:
        print(f'{user_name} requested: {query} in {details[0]["title"]}')
        await send_cb(content=f"Looking through {details[0]['title'] if details else 'everywhere'}. This may take a while!")
        # only keep BATCHSIZE concurrent requests!
        pbar = tqdm(total=BATCHSIZE, desc="queue progress", leave=False)
        pbar.n = BATCHSIZE
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            if len(yes_token_ids) == 0:
                yes_token_id = await get_tok(session, yes_tokens)
            if len(no_token_ids) == 0:
                no_token_id = await get_tok(session, no_tokens)

            num_hunks = 8
            producer = generate_cb(details)
            scores = await process(
                producer, pbar, session, query,
                book_sep, yes_token_id, no_token_id,
                num_hunks)
            pbar.close()

            returned_limit = 5

            print(f'Scores accumulated. Sending Best {len(scores)}')
            no_results = True
            for selection in scores:
                if selection["score"] < 0.875:
                    break
                await asyncio.sleep(0.5)
                # server load protection, otherwise llama.cpp kicks
                num_verses = 3
                producer = get_tasks_for_selection(generate_cb, selection)
                pbar = tqdm(total=BATCHSIZE,
                            desc="queue progress", leave=False)
                pbar.n = BATCHSIZE
                scores = await process(
                    producer, pbar, session, query,
                    book_sep, yes_token_id, no_token_ids,
                    num_verses)
                for best_verse in scores:
                    if best_verse["score"] < 0.5:
                        break
                    no_results = False
                    returned_limit -= 1
                    if not returned_limit:
                        break
                    bs4text = BeautifulSoup(best_verse['verse'], features="html.parser").get_text()
                    await asyncio.sleep(0.5)
                    # rate limit the verse output
                    await send_safe(
                        interaction,
                        f"""in chunk: {selection['ref']}, score {get_score(selection)}
THIS PARTICULAR VERSE: ref: {best_verse['ref']}, score: {get_score(best_verse)}, {bs4text}""")
                pbar.close()
                if not returned_limit:
                    break
            if no_results or not scores:
                print('no good results')
                await send_safe(interaction, 'nothing relevant.')
            else:
                await send_safe(interaction, "no further results.")
        print(f'finished request for {user_name}.')
    except Exception as e:
        await do_error(interaction, e)
        raise e
    finally:
        await timeoutLock.finished()


async def validate_bible(interaction, query, normname, query_user, book_name_user, user_name):
    send_cb = interaction.edit_original_response
    for book, variants in zip(
        KJV_BOOK_DETAILS,
        [['gen'], ['exo'], ['lev'], ['num'], ['deu'], ['jos'], ['judg', 'jug'], ['rut'], ['1sa', '1 sa'], ['2sa', '2 sa'], ['1ki', '1 ki'], ['2ki', '2 ki'],
         ['1ch', '1 ch'], ['2ch', '2 ch'], ['ezr'], ['neh'], ['est'], ['job'], ['psa'], ['pro'], ['ecc'], ['son'], ['isa'], ['jer'], ['lam'], ['eze'], ['dan'],
         ['hos'], ['joe'], ['amo'], ['oba'], ['jon'], ['mic'], ['nah'], ['hab'], ['zep'], ['hag'], ['zec'], ['mal'],
         ['mat'], ['mar'], ['luk'], ['joh'], ['act'], ['rom'], ['1co', '1 co'], ['2co', '2 co'], ['gal'], ['eph'], ['phi'],
         ['col'], ['1th', '1 th'], ['2th', '2 th'], ['1ti', '1 ti'], ['2ti', '2 ti'], ['tit'], ['phi'], ['heb'], ['jam'],
         ['1pe', '1 pe'], ['2pe', '2 pe'], ['1jo', '1 jo'], ['2jo', '2 jo'], ['3jo', '3 jo'], ['jude'], ['rev']]
    ):
        if any(normname.startswith(v) for v in variants):
            selectedbook = book
            break
    else:
        await send_cb(content=f'Error: I do not recognise the book: "{book_name_user}" {user_name}.')
        return
    # validate args:
    base_len = len(get_data(query, ""))
    if (base_len > 400):
        await send_cb(content=f'Error: Query result is {base_len-400} characters over the limit. Please refine your search, {user_name}.')
        return
    return [{"book_id": selectedbook, "title": selectedbook}]


async def validate_egw(interaction, query, normname, query_user, book_name_user, user_name):
    send_cb = interaction.edit_original_response
    base_len = len(get_data(query, ""))
    if (base_len > 400):
        await send_cb(content=f'Error: Query result is {base_len-400} characters over the limit. Please refine your search, {user_name}.')
        return
    book_id = None
    try:
        book_id = int(normname)
    except ValueError:
        pass
    if book_id is None:
        candidates = []
        for elem in EGW_BOOK_DETAILS:
            elemcodes = elem['code'].lower().split('/')
            if normname in elemcodes:
                return [elem]
            elemtitle = normalize_string(elem['title'])
            if normname == elemtitle:
                return [elem]
            if normname in elemtitle:
                candidates.append((elem, 1))
        if candidates:
            book_id = await choose_candidates(interaction, candidates)
    if book_id is None:
        candidates = []
        for elem in EGW_BOOK_DETAILS:
            elemtitle = set(normalize_string(elem['title']).split(' '))
            query = set(normname.split(' '))
            matches = len(query & elemtitle)
            if matches > 0:
                candidates.append(
                    (elem, matches + 1 / (len(query) - matches + 1)))
        book_id = await choose_candidates(interaction, candidates)
    if book_id is None:
        await send_safe(interaction, 'Sorry! no further matching method!')
        return
    details = [elem for elem in EGW_BOOK_DETAILS if str(elem['book_id']) == str(book_id)]
    if len(details) < 1:
        await send_safe(interaction, f'Error: Book id not found, {user_name}.')
        return
    return details[:1]


async def do_search_validate(interaction, generate_cb, validate_cb, book_sep, user_name, book_name_user: str, query_user: str):
    global yes_token_ids, no_token_ids, timeout_value
    send_cb = interaction.edit_original_response
    should_obey_timeout = str(user_name) not in ['micsthepick']
    # todo better whitelist (don't use str(user_name))!

    has_acquired_timeout = await timeoutLock.try_acquire()
    if not has_acquired_timeout:
        await send_cb(content=f"Wait for the current request to finish first, {user_name}!")
        return

    can_start_in = await timeoutLock.can_start_in()
    if can_start_in is not None:
        if should_obey_timeout:
            remaining = int(ceil(max(can_start_in.total_seconds(), 0)))
            await send_cb(content=f'Please wait {remaining} seconds before trying again, {user_name}')
        else:
            await send_cb(content=f'{user_name} is bypassing the current timeout!')
    try:
        async with processingSem:
            # interaction is the context object automatically passed by Twitchio
            # Parse the command message
            # parse args
            query = normalize_string(query_user)
            normname = normalize_string(book_name_user)

            details = await validate_cb(interaction, query, normname, query_user, book_name_user, user_name)

            if details is None:
                await timeoutLock.finished()
                return

            await do_search(interaction, generate_cb, book_sep, user_name, query, details)
            await timeoutLock.start()
    except Exception as e:
        await timeoutLock.finished()
        await do_error(interaction, e)
        raise e


# Command to handle searching
@bot.tree.command(name='search')
async def search(interaction, book_name: str, query: str):
    """Search one book of the bible.
    usage: !search <book> <search query>"""
    user = interaction.user

    await interaction.response.send_message("Please wait...")

    await do_search_validate(interaction, bible_gen_cb, validate_bible, ':', user, book_name, query)


@bot.tree.command(name='searchegw')
async def searchegw(interaction, book_name: str, query: str):
    """Search inspired writings.
    usage: !search <id OR book code OR ~title>"""
    user = interaction.user

    await interaction.response.send_message("Please wait...")

    await do_search_validate(interaction, egw_gen_cb, validate_egw, '.', user, book_name, query)


@bot.event
async def on_ready():
    await bot.tree.sync()  # Synchronize commands with Discord
    print(f'Logged in as {bot.user}')

try:
    kjv_details_path = './Bible-kjv/Books.json'
    KJV_BOOK_DETAILS = asyncio.run(load_books(kjv_details_path))
except FileNotFoundError:
    print(f"Error: The file {kjv_details_path} does not exist.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: The file {kjv_details_path} is not a valid JSON file.")
    sys.exit(1)

try:
    egw_details_path = './egwbooks/books.json'
    EGW_BOOK_DETAILS = asyncio.run(load_books(egw_details_path))
except FileNotFoundError:
    print(f"Error: The file {egw_details_path} does not exist. "
          "(have you imported the .egwbooks and run egw_extractor.py?)")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: The file {egw_details_path} is not a valid JSON file.")
    sys.exit(1)

# Run the bot
if __name__ == '__main__':
    bot.run(token=bot_token)
