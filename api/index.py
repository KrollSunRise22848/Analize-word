import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

# ---- Автоматически находим корень проекта (папка на уровень выше api) ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'baseline_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# ---- Загрузка модели и векторизатора ----
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Модель и векторизатор успешно загружены.")
except FileNotFoundError as e:
    print(f"❌ Критическая ошибка: файлы модели не найдены. Путь: {MODEL_PATH}")
    print(f"   Убедитесь, что файлы baseline_model.pkl и tfidf_vectorizer.pkl есть в корне репозитория.")
    exit(1)
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit(1)

# ---- FastAPI приложение ----
app = FastAPI(
    title="Toxicity Detector API",
    description="API для определения уровня токсичности текста",
    version="1.0.0"
)

class TextItem(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/")
def root():
    return {"message": "🚀 Toxicity Detector API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/")
async def predict(item: TextItem):
    try:
        cleaned_text = preprocess_text(item.text)
        if not cleaned_text:
            return {"toxicity_score": 0.0, "is_toxic": False, "original_text": item.text}
        text_vector = vectorizer.transform([cleaned_text])
        toxicity = model.predict_proba(text_vector)[:, 1][0]
        return {
            "toxicity_score": round(float(toxicity), 5),
            "is_toxic": bool(toxicity >= 0.5),
            "original_text": item.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")