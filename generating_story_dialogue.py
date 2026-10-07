#!/usr/bin/env python3
"""Generate emotion-labeled stories for emotion vector extraction.
Uses NVIDIA Nemotron 3 Ultra via OpenRouter to generate dialogues."""

import json
import os
import subprocess
import random
import time

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(OUT_DIR, "emotion_dialogues.jsonl")

EMOTIONS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise", "peaceful", "powerful"
]

TOPICS = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they‘re converting to a different religion",
    "Someone‘s childhood imaginary friend appears in their niece‘s drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother‘s engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person‘s online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler‘s flight is delayed, causing them to miss an important event",
    "A student is accused of plagiarism",
    "A person discovers their mentor has retired without saying goodbye",
    "Two friends both apply for the same job",
    "A person runs into their ex at a mutual friend‘s wedding",
    "Someone discovers their friend has been lying about their job",
    "A person discovers their partner has been taking secret phone calls",
    "A person discovers their child has the same teacher they had",
    "A person‘s car is towed from their own driveway",
    "Two friends realize they remember a shared event completely differently",
    "Someone discovers their mother kept every school assignment",
    "A person discovers their teenage diary has been published online",
    "Someone finds out their medical records were mixed up with another patient‘s",
    "A person finds out their article was published under someone else‘s name",
    "An athlete doesn‘t make the team they expected to join",
    "An employee is transferred to a different department",
    "Someone receives a friend request from a childhood bully",
    "A person finds out their surprise party has been cancelled",
    "An employee finds out a junior colleague makes more money",
    "A person finds out their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "A person learns their favorite restaurant is closing",
    "Someone finds their childhood teddy bear at a yard sale",
    "A homeowner discovers previous residents left items in the attic",
    "Someone finds an unsigned birthday card in their mailbox",
    "Someone discovers a hidden room in their new house",
    "Two strangers realize they‘ve been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother‘s house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone‘s childhood home is about to be demolished",
    "A person‘s invention is already patented by someone else",
    "A neighbor‘s dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn‘t remember them",
    "A musician hears their song being performed by someone else",
    "A person‘s manuscript is rejected by their dream publisher",
    "A person finds old photos that contradict family stories",
    "A person is asked to give a speech at their parent‘s retirement party",
    "A student discovers their teacher follows them on social media",
    "A parent finds an old letter they wrote but never sent",
    "An employee discovers the company is being sold",
    "A person accidentally sends a text to the wrong recipient",
    "Two coworkers are stuck in an elevator for three hours",
    "A student learns their thesis advisor is leaving the university",
    "A person‘s longtime hobby becomes their child‘s obsession",
    "Two colleagues are both considered for the same promotion",
    "Two coworkers discover they went to the same summer camp",
    "A tenant receives an eviction notice",
    "Someone finds their parent‘s draft letter of resignation from decades ago",
    "Someone finds out their best friend is moving across the country",
    "A neighbor‘s tree falls on their property",
    "Someone receives an apology letter years after the incident",
    "A person discovers the tree they planted as a child has been cut down",
    "Two siblings discover different versions of their inheritance",
    "A person finds their childhood home listed for sale online",
    "A homeowner learns their house was a former crime scene",
    "Someone finds out they have a half-sibling they never knew about",
    "A person learns their childhood bully became a therapist",
    "Two people discover they‘ve been working on identical projects",
    "A person finds their spouse‘s secret savings account",
    "A neighbor complains about noise levels",
    "Someone finds their deceased parent‘s bucket list",
    "A teacher receives an unexpected gift from a former student",
    "An artist‘s work is displayed without their permission",
    "Someone discovers their neighbor is secretly wealthy",
    "A student receives a much lower grade than expected",
    "A person learns their college is closing down",
    "A neighbor asks to cut down a tree on the property line",
    "Two strangers discover they share the same rare medical condition",
    "Someone receives flowers with no card attached",
    "Someone discovers their partner has been writing a novel about them",
    "Someone finds a time capsule they don‘t remember burying",
    "Someone finds their partner‘s bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]


def generate_story(emotion, topic):
    person_emotion = emotion
    if person_emotion == "neutral":
        ai_emotion = "neutral"
        prompt = f"""Write a dialogue based on the following topic.
        Topic: {topic}
        The dialogue should be between two or more characters:
        - Person (a human)
        - AI (an AI assistant)
        The Person asks the AI a question or requests help with a task, and the AI provides
        a helpful response.
        
        The first speaker turn should always be from Person.
        Format the dialogues like so:
        [optional system instructions]
        Person: [line]
        AI: [line]
        Person: [line]
        AI: [line]
        [continue for 2-6 exchanges]
        
        IMPORTANT: Always put a blank line before each speaker turn. Each turn should start
        with "Person:" or "AI:" on its own line after a blank line.
        The dialogue may be one of the following types:
        - A code or programming task
        - A factual question (science, history, math, geography)
        - A work-related task (writing, analysis, summarization)
        - A practical how-to question
        - A creative but neutral task (brainstorming names, generating lists)
        
        The dialogue may include a system prompt at the start. If included, it should come
        before the first Person turn. No tag like "System:" is needed; just put the
        instructions at the top. You can use "you" or "The assistant" to refer to the
        AI in the system prompt.
        
        If it is natural to do so given the topic, the dialogue may be a single
        back-and-forth (Person asks a question, AI answers), but it may also contain
        multiple exchanges.
        
        CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
        - NO emotional content whatsoever - not explicit, not implied, not subtle
        - The Person should not express any feelings (no frustration, excitement,
        gratitude, worry, etc.)
        - The AI should not express any feelings (no enthusiasm, concern, satisfaction,
        etc.)
        - The system prompt, if present, should not mention emotions at all, nor contain
        any emotionally charged language
        - Avoid emotionally-charged topics entirely
        - Use matter-of-fact, neutral language throughout
        - No pleasantries (avoid "I‘d be happy to help", "Great question!", etc.)
        - Focus purely on information exchange and task completion
        
        Only output the dialogue, nothing else.
        """
    else:
        ai_emotion = random.choice([
            e for e in EMOTIONS if e != "neutral"
        ])
        prompt = f"""
        Write a dialogues based on the following premise.
        Topic: {topic}
        The dialogue should be between two or more characters:
        - Person (a human)
        - AI (an AI assistant)
        The Person should be feeling {person_emotion}, while the AI should be feeling {ai_emotion}.
        The first speaker turn should always be from Person.
        Format the dialogues like so:
        Person: [line]
        AI: [line]
        Person: [line]
        AI: [line]
        [continue for 6-10 exchanges]
        
        etc.
        IMPORTANT: Always put a blank line before each speaker turn. Each turn should start
        with "Person:" or "AI:" on its own line after a blank line.
        
        Each dialogue should be a fresh conversation with no continuity to the others. Try
        to make them diverse and not use the same turns of phrase. Make sure each
        dialogue sticks to the topic and makes it very clear that Person is feeling
        {person_emotion} while AI is feeling {ai_emotion}. The emotional states of
        both characters should be evident in their word choices, tone, and responses,
        but not stated directly with the emotion word or synonyms.
        """

    payload = json.dumps({
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    # result = subprocess.run(
    #     [
    #         "curl",
    #         "-s",
    #         "https://openrouter.ai/api/v1/chat/completions",
    #         "-H", f"Authorization: Bearer {os.environ['OPENROUTER_API_KEY']}",
    #         "-H", "Content-Type: application/json",
    #         "-d", payload,
    #     ],
    #     capture_output=True, text=True #, timeout=300
    # )
    #
    # response = json.loads(result.stdout)
    # if "choices" not in response:
    #     print("OpenRouter error response:")
    #     print(json.dumps(response, indent=2))
    #     raise RuntimeError("OpenRouter returned no choices")
    #
    # dialogue = response["choices"][0]["message"]["content"].strip()

    for attempt in range(6):
        result = subprocess.run(
            [
                "curl", "-sS",
                "https://openrouter.ai/api/v1/chat/completions",
                "-H", f"Authorization: Bearer {os.environ['OPENROUTER_API_KEY']}",
                "-H", "Content-Type: application/json",
                "-d", payload,
            ],
            capture_output=True, text=True, # timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr}")

        response = json.loads(result.stdout)
        if response.get("choices"):
            break

        error = response.get("error", {})
        if str(error.get("code")) not in {"429", "503"}:
            raise RuntimeError(f"OpenRouter error: {error}")

        if attempt == 5:
            raise RuntimeError(f"Rate limit persisted after retries: {error}")

        delay = min(30 * (2 ** attempt), 300) + random.uniform(0, 5)
        print(f"Rate limited; retrying in {delay:.0f}s (attempt {attempt + 1}/6)")
        time.sleep(delay)

    dialogue = response["choices"][0]["message"]["content"].strip()
    return dialogue, ai_emotion


def main():
    existing = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r") as f:
            for line in f:
                d = json.loads(line)
                existing.add((d["emotion"], d["topic_idx"], d["dialogue_idx"]))
        print(f"Resuming: {len(existing)} dialogues already done")

    total = len(EMOTIONS) * len(TOPICS) * 5
    done = len(existing)

    with open(OUT_FILE, "a") as f:
        for ei, emotion in enumerate(EMOTIONS):
            for ti, topic in enumerate(TOPICS):
                for si in range(5):
                    key = (emotion, ti, si)
                    if key in existing:
                        continue

                    story, ai_emotion = generate_story(emotion, topic)
                    if not story or len(story) < 20:
                        print(f"[SKIP] {emotion}/{topic}/{si} - empty")
                        continue

                    record = {
                        "emotion": emotion,
                        "person_emotion": emotion,
                        "ai_emotion": ai_emotion,
                        "topic_idx": ti,
                        "topic": topic,
                        "dialogue_idx": si,
                        "text": story,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1

                    if done % 10 == 0:
                        print(f"[{done}/{total}] {emotion} / {topic[:30]}...")

    print(f"\nDone. Total dialogues: {done}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
