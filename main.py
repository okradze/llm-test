import asyncio
import os
import json
import html
import re
import unicodedata

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompts import LinkContentPrompt

load_dotenv()

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")

if not OPEN_ROUTER_API_KEY:
    print("OPEN_ROUTER_API_KEY is not set")
    exit()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)


# Define models by provider
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "qwen": [
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-2.5-72b-instruct",
    ],
    "deepseek": [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1-distill-qwen-1.5b",
        "deepseek/deepseek-r1-distill-qwen-14b",
        "deepseek/deepseek-r1-distill-qwen-32b",
        "deepseek/deepseek-r1-distill-llama-70b",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
    ],
    "minstral": [
        "mistralai/mistral-small-24b-instruct-2501",
        "mistralai/ministral-8b",
    ],
}


# Cache for OpenAI responses
openai_cache = {}  # Structure: {tweet_id: {model: result}}


async def score_by_model(messages, model: str, tweet_id: str):
    if tweet_id and "gpt" in model:
        if tweet_id in openai_cache and model in openai_cache[tweet_id]:
            return openai_cache[tweet_id][model]

    completion = await client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        timeout=300,
        temperature=0.0001,
        top_p=0.0001,
    )

    content = completion.choices[0].message.content
    extracted_score = None

    if isinstance(content, str):
        extracted_score = int(LinkContentPrompt().extract_score(content))

    result = {"score": extracted_score, "result": content}

    if "gpt" in model:
        openai_cache[tweet_id] = openai_cache.get(tweet_id, {})
        openai_cache[tweet_id][model] = result

    return result


async def score_item(message, models, item):
    id = item.get("tweet_id")

    scoring_tasks = []

    # Create scoring tasks for all models
    for provider, provider_models in models.items():
        for model in provider_models:
            scoring_tasks.append(
                asyncio.create_task(score_by_model(message, model, id))
            )

    # Execute all scoring tasks concurrently
    results = await asyncio.gather(*scoring_tasks)

    # Create a dictionary mapping model names to their results
    model_results = {}
    model_index = 0

    for provider, provider_models in models.items():
        for model in provider_models:
            model_results[model] = results[model_index]
            model_index += 1

    return model_results, item, message


def get_scoring_text(prompt: str, content: str):
    try:
        scoring_prompt = LinkContentPrompt()
        content = clean_text(content)
        scoring_prompt_text = scoring_prompt.text(prompt, content)

        return [
            {"role": "system", "content": scoring_prompt.get_system_message()},
            {"role": "user", "content": scoring_prompt_text},
        ]
    except Exception as e:
        print(e)
        return None


def clean_text(text):
    # Unescape HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)

    # Remove mentions at the beginning of the text
    text = re.sub(r"^(@\w+\s*)+", "", text)

    # Remove emojis and other symbols
    text = re.sub(r"[^\w\s,]", "", text)

    # Normalize whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Remove non-printable characters and other special Unicode characters
    text = "".join(
        char
        for char in text
        if char.isprintable() and not unicodedata.category(char).startswith("C")
    )

    return text


async def process_items_group(batch_items, models):
    tasks = []

    for item in batch_items:
        prompt = item.get("question")
        tweet_text = item.get("tweet_text")
        messages = get_scoring_text(prompt=prompt, content=tweet_text)
        tasks.append(asyncio.create_task(score_item(messages, models, item)))

    results = []

    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        results.append(result)
        print(f"Completed {i}/{len(tasks)} tasks")

    items = []
    scores = {model: {} for provider in models.values() for model in provider}
    matches = {
        model: 0
        for provider in models.values()
        for model in provider
        if model != "gpt-4o-mini"
    }

    for model_results, item, messages in results:
        processed_item = {
            "question": item.get("question"),
            "tweet_id": item.get("tweet_id"),
            "tweet_text": item.get("tweet_text"),
            "prompt": f"{messages[0]['content']}\n{messages[1]['content']}",
        }

        # Get gpt-4o-mini score for comparison
        mini_score = model_results.get("gpt-4o-mini", {}).get("score")

        # Add model results to the item and count matches
        for model, result in model_results.items():
            processed_item[model] = result

            # Update score statistics
            score = result["score"]
            if score is not None:
                scores[model][score] = scores[model].get(score, 0) + 1

            # Count matches with gpt-4o-mini
            if (
                model != "gpt-4o-mini"
                and score == mini_score
                and mini_score is not None
            ):
                matches[model] += 1

        items.append(processed_item)

    return {"scores": scores, "matches": matches, "total": len(results), "items": items}


async def main():
    # Read data.json
    with open("data.json", "r") as f:
        try:
            items = json.loads(f.read())[:100]  # Limited to 10 items for testing
        except json.JSONDecodeError:
            print("Error reading data.json")
            return

    GROUP_SIZE = 4

    all_items = []
    all_scores = {model: {} for provider in MODELS.values() for model in provider}
    all_matches = {
        model: 0
        for provider in MODELS.values()
        for model in provider
        if model != "gpt-4o-mini"
    }

    # Process all groups
    for i in range(0, len(items), GROUP_SIZE):
        batch_items = items[i : i + GROUP_SIZE]
        group_results = await process_items_group(batch_items, MODELS)

        all_items.extend(group_results["items"])

        # Merge scores
        for model in all_scores:
            for score, count in group_results["scores"][model].items():
                all_scores[model][score] = all_scores[model].get(score, 0) + count

        # Accumulate matches
        for model, matches in group_results["matches"].items():
            all_matches[model] += matches

        print(f"Processed group {i + 1} - {i + GROUP_SIZE}")

    # Sort scores in ascending order for each model
    sorted_scores = {
        model: dict(sorted(all_scores[model].items())) for model in all_scores
    }

    sorted_matches = dict(
        sorted(all_matches.items(), key=lambda item: item[1], reverse=True)
    )

    # Save results
    results_json = {
        "result": {
            "total": len(all_items),
            "matches": sorted_matches,
            "scores": sorted_scores,
        },
        "items": all_items,
    }

    with open("result.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Processing complete. Total items processed: {len(all_items)}")


if __name__ == "__main__":
    asyncio.run(main())
