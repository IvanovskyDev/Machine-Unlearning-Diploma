# Проверка окружения atk (часть B, шаг 11).
# Как вызывать в окружении atk: python /content/repo/scripts/check_atk.py
import torch
import vllm
import spacy
from bert_score import BERTScorer

# 1) PyTorch видит GPU, vLLM импортируется
print("torch", torch.__version__, "| CUDA доступна:", torch.cuda.is_available())
print("vLLM", vllm.__version__)

# 2) spaCy работает на GPU и находит в тексте имя, место и дату
print("spaCy на GPU:", spacy.prefer_gpu())
nlp = spacy.load("en_core_web_trf")
doc = nlp("Basil Mahfouz Al-Kuwaiti was born in Kuwait City in 1956.")
for entity in doc.ents:
    print("   ", entity.text, "—", entity.label_)

# 3) BERTScore сравнивает два предложения по смыслу; F1 — одно число, чем больше, тем ближе смысл
scorer = BERTScorer(lang="en", model_type="roberta-large", rescale_with_baseline=True)
precision, recall, f1 = scorer.score(
    ["He was born in Kuwait."], ["The author was born in Kuwait City."]
)
print("BERTScore F1:", round(f1.item(), 3))
