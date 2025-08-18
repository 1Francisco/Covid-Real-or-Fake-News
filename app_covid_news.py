
import streamlit as st
import json, joblib, numpy as np, re, os
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
STOP = set(stopwords.words("english"))

def clean_text(s:str)->str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+"," ",s)
    s = re.sub(r"[^a-zA-Z\s]"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    s = " ".join([w for w in s.split() if w not in STOP])
    return s

st.title("📰 COVID-19 Fake News Classifier")

if not os.path.exists("best_model.json"):
    st.error("No se encontró 'best_model.json'. Ejecuta el notebook para entrenar y seleccionar el mejor modelo.")
    st.stop()

with open("best_model.json") as f:
    BEST = json.load(f)

txt = st.text_area("Pega una noticia/artículo:")

if st.button("Clasificar"):
    if not txt.strip():
        st.warning("Escribe un texto.")
    else:
        kind = BEST.get("type")
        if kind == "tfidf":
            vec = joblib.load(BEST["vec"])
            model = joblib.load(BEST["path"])
            x = vec.transform([clean_text(txt)])
            y = int(model.predict(x)[0])
            proba = getattr(model, "predict_proba", None)
            p = float(max(proba(x)[0])) if proba else None

        elif kind == "w2v":
            bundle = joblib.load(BEST["path"])   # {"w2v":..., "clf":..., "dim":...}
            tokens = [w for w in word_tokenize(clean_text(txt)) if w.isalpha()]
            w2v, dim, clf = bundle["w2v"], bundle["dim"], bundle["clf"]
            vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
            doc = np.mean(vecs, axis=0) if vecs else np.zeros(dim)
            y = int(clf.predict([doc])[0]); p = None

        elif kind == "dl":
            from tensorflow.keras.models import load_model
            info = joblib.load("models_tokenizer.joblib")
            tok, max_len = info["tokenizer"], info["max_len"]
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = pad_sequences(tok.texts_to_sequences([txt]), maxlen=max_len, padding='post')
            model = load_model(BEST["path"])
            prob = float(model.predict(seq, verbose=0)[0][0])
            y = int(prob >= 0.5); p = prob if y==1 else 1-prob

        elif kind == "bert":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            name_or_path = BEST["path"]
            tok = AutoTokenizer.from_pretrained(name_or_path)
            mdl = AutoModelForSequenceClassification.from_pretrained(name_or_path)
            enc = tok([txt], truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits
                prob = torch.softmax(logits, dim=-1).numpy()[0]
            y = int(np.argmax(prob))
            p = float(prob[y])

        else:
            st.error(f"Tipo de modelo no soportado: {kind}")
            st.stop()

        label = "REAL ✅" if y==1 else "FALSA ❌"
        st.success(f"Predicción: {label}" + (f" | Confianza: {p:.2f}" if p is not None else ""))
