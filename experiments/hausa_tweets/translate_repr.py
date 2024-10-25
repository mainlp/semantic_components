from semantic_components.translate_gpt import translate_phrase, translate_list_of_tokens
import pandas as pd
import pickle
import time

# load the representations
df = pd.read_pickle("results/sca_normco_N=377789_mu=1.0_mcs=300_ms=300_ng/representations_translated.pkl")
df.rename(columns={"representation_metroid": "representation_medoid"}, inplace=True)

df.head()

price_in_tok_mini = 0.125 / 1000000
price_out_tok_mini = 0.6 / 1000000
price_in_tok = 2.5 / 1000000
price_out_tok = 10 / 1000000

translations = []
responses = []

n_in_toks = 0
n_out_toks = 0

# translate the medoids
for i, m in enumerate(df["representation_medoid"]):
    if i < len(translations):
        continue
    translation, response = None, None
    while translation is None:
        print(i)
        try:
            translation, response = translate_phrase(m, lang="a Nigerian language", model="gpt-4o-mini")
        except Exception as e:
            print("Rate limit reached, retrying in 30 seconds")
            print(e)
            time.sleep(30)
    print("Translated '", m, "'to '" + translation + "'")
    translations.append(translation)
    responses.append(response)
    n_in_toks += response.usage.prompt_tokens
    n_out_toks += response.usage.completion_tokens
    print("Current prompt tokens:", n_in_toks, "Current completion tokens:", n_out_toks, "Cost:", 
          n_in_toks*price_in_tok_mini + n_out_toks*price_out_tok_mini, "USD (mini)",
            n_in_toks*price_in_tok + n_out_toks*price_out_tok, "USD (standard)")
    print()


df["representation_medoid_en_mini"] = translations

df.head()

# save responses and translations
df.to_pickle("results/sca_normco_N=377789_mu=1.0_mcs=300_ms=300_ng/representations_en.pkl")
with open(
    "results/sca_normco_N=377789_mu=1.0_mcs=300_ms=300_ng/representations_en_responses_medoids_mini.pkl", "wb") as f:
    pickle.dump(responses, f)

translations_tokens = []
responses_tokens = []

# translate the tokens
for tokens in df["representation"]:
    translation, response = translate_list_of_tokens(tokens, lang="a Nigerian language", model="gpt-4o-mini")
    translations_tokens.append(translation)
    responses_tokens.append(response)
    print("Translated", tokens, "to '" + translation + "'")
    n_in_toks += response.usage.prompt_tokens
    n_out_toks += response.usage.completion_tokens
    print("Current prompt tokens:", n_in_toks, "Current completion tokens:", n_out_toks, "Cost:", 
          n_in_toks*price_in_tok_mini + n_out_toks*price_out_tok_mini, "USD (mini)", 
          n_in_toks*price_in_tok + n_out_toks*price_out_tok, "USD (standard)")

df["representation_en_mini"] = translations_tokens

# save responses and translations
df.to_pickle("results/sca_normco_N=377789_mu=1.0_mcs=300_ms=300_ng/representations_translated.pkl")
with open(
    "results/sca_normco_N=377789_mu=1.0_mcs=300_ms=300_ng/representations_en_responses_tokens_mini.pkl", "wb") as f:
    pickle.dump(responses_tokens, f)

df.head()