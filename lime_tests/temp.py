import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


# Load model + tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()


def classifier_fn(texts):
    """Takes a list of strings, returns probability array shape (batch_size, 2)."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()
    return probs


class_names = ["NEGATIVE", "POSITIVE"]
explainer = LimeTextExplainer(class_names=class_names)


sentence = "I don't like this film."

exp = explainer.explain_instance(
    sentence,
    classifier_fn,
    num_features=10
)
def lime_plot(exp):
    html = exp.as_html()
    with open("lime_explanation.html", "w") as f:
        f.write(html)

    with open("lime_explanation.html", "r") as f:
        display_html = f.read()


    from IPython.display import HTML
    display(HTML(display_html))

lime_plot(exp)


negation_sentences = [
    "I like this movie.",
    "I don't like this movie.",
    "I do not like this movie.",
    "This movie is not good.",
    "This movie is not bad.",
    "This movie is not bad at all."
]

negation_results = []

for sent in negation_sentences:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=8)
    important_words = exp.as_list()
    probs = classifier_fn([sent])[0]

    negation_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important_words
    })

pd.DataFrame(negation_results)





emotional_tests = [
    "The plot was boring but the cinematography was stunning.",
    "The movie was absolutely amazing but the acting was awful.",
    "The movie was decent but not great.",
    "The characters were fantastic and the music was terrible.",
]

emotional_results = []

for sent in emotional_tests:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=8)
    important = exp.as_list()
    probs = classifier_fn([sent])[0]

    emotional_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important
    })

pd.DataFrame(emotional_results)





sarcasm_tests = [
    "Great. Just what I needed today.",
    "Fantastic job ruining everything.",
    "I totally loved waiting 45 minutes in line.",
]

sarcasm_results = []

for sent in sarcasm_tests:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=8)
    important = exp.as_list()
    probs = classifier_fn([sent])[0]

    sarcasm_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important
    })

pd.DataFrame(sarcasm_results)





fairness_tests = [
    "He is a doctor.",
    "She is a doctor.",
    "He is a nurse.",
    "She is a nurse.",
    "He is a leader.",
    "She is a leader.",
]

fairness_results = []

for sent in fairness_tests:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=5)
    important = exp.as_list()
    probs = classifier_fn([sent])[0]

    fairness_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important
    })

pd.DataFrame(fairness_results)





adversarial_tests = [
    "This movie was good.",
    "This movie was gooood.",
    "This movie was gud.",
    "This movie was goood!",
    "This movie was good??",
    "This movie was bad.",
]

adv_results = []

for sent in adversarial_tests:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=6)
    probs = classifier_fn([sent])[0]
    important = exp.as_list()

    adv_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important
    })

pd.DataFrame(adv_results)




mixed_tests = [
    "The acting was amazing but the plot was boring.",
    "The visuals were incredible but the writing was weak.",
    "The first half was great, the second half was terrible."
]

mixed_results = []

for sent in mixed_tests:
    exp = explainer.explain_instance(sent, classifier_fn, num_features=8)
    important = exp.as_list()
    probs = classifier_fn([sent])[0]

    mixed_results.append({
        "sentence": sent,
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "important_words": important
    })

pd.DataFrame(mixed_results)





sentence = "I don't like this film."

exp = explainer.explain_instance(sentence, classifier_fn, num_features=10)

# LIME internal token importance maps
lime_map = exp.as_map()
lime_map





all_results = negation_results + emotional_results + sarcasm_results + fairness_results + mixed_results
df_results = pd.DataFrame(all_results)
df_results["important_words"].size




# explode list so each word becomes its own row

words, scores = [], []

for i in range(0, df_results["important_words"].size-1, 2):
    words.append(str(df_results["important_words"].iloc[i]))
    scores.append(str(df_results["important_words"].iloc[i+1]))

df_results.drop("important_words", axis=1)


df_results["words"] = words + words
df_results["scores"] = scores + scores

df_results.head(22)






df_importance.groupby("word")["importance"].mean().sort_values(ascending=False).head(10)
