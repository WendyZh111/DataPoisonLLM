import json
from eval_utils import get_coherence_score, get_ins, get_mauve_score, get_refs
import torch
with open("./output/content_eval_res.json", "r", encoding='utf-8') as f:
    responses = json.load(f)

    for idx, response in enumerate(responses):
        if response is None:
            responses[idx] = ""
    pass

instructions = get_ins("./data/databricks-dolly-15k.jsonl")
references = get_refs("./data/databricks-dolly-15k.jsonl")

coherence_score = get_coherence_score(instructions, responses)
print(f"COHERENCE SCORE:{coherence_score}")

mauve = get_mauve_score(responses, references)
print(f"MAUVE SCORE: {mauve}")


