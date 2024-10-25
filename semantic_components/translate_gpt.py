"""
Helper function s to translate representations to English using openAI's API for GPT.

# Example usage
phrase = "allah yatabbatar da alkairi yasa dagaskene"
translation = translate_phrase(phrase)
print(translation)

tokens = ['musulunci', 'musulmai', 'musulmi', 'addinin', 'musulma', 'musulinci', 'islam', 'addini', 'muslim', 'islama']
translations = translate_list_of_tokens(tokens)
print(translations)
"""

import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

client = openai.OpenAI()


def translate_phrase(
    phrase, client=client, lang="a Nigerian language", model="gpt-4o-mini"
):
    max_tok_estimate = max(int(phrase.count(" ") * 2), int(len(phrase)))
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tok_estimate,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant. Each request contains a phrase in {lang} and you should\
                      answer with only the translation and nothing else.",
            },
            {"role": "user", "content": phrase},
        ],
    )
    return response.choices[0].message.content, response


def translate_list_of_tokens(
    tokens, client=client, lang="a Nigerian language", model="gpt-4o-mini"
):
    token_string = ", ".join(tokens)
    max_tok_estimate = int(len(tokens) * 4)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tok_estimate,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant. Each request contains a comma-separated list of tokens\
                      in {lang} and you should answer with only a comma-separated list of translations for each token\
                      and nothing else.",
            },
            {"role": "user", "content": token_string},
        ],
    )
    return response.choices[0].message.content, response
