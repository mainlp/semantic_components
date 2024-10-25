# translate metroids and tokens
# load afro xlmr

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device available for running: ", device)

# load representations
experiment_name = "nigerian_tweets"
representations = pd.read_pickle(f"results/{experiment_name}.pkl")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to(device)

translator = pipeline(
    "translation", 
    model=model,
    tokenizer=tokenizer, 
    src_lang="hau_Latn", 
    tgt_lang="eng_Latn",
    device=device,
    max_length=300
)

sentence = "to tsugunne bata kare ba kenan"

translated = translator(sentence)

print(translated)


# get dataloader of metroids for batching
metroids = representations["metroids"]
dataloader = torch.utils.data.DataLoader(metroids, batch_size=128, shuffle=False)

translated = []
for i, batch in enumerate(dataloader):
    print("Batch: ", i)
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    print(len(outputs.hidden_states))
    print(outputs.hidden_states[-2].shape)
    translated.append(outputs.logits.cpu().numpy())

