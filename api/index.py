import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

# 1. Загружаем модель и векторизатор при старте сервера
try:
    with open('../baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Модель и векторизатор успешно загружены.")
except FileNotFoundError:
    print("❌ Критическая ошибка: файлы модели не найдены.")
    exit()
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit()

# 2. Создаем приложение FastAPI
app = FastAPI(
    title="Toxicity Detector API",
    description="API для определения уровня токсичности текста на основе предобученной модели",
    version="1.0.0"
)

# 3. Определяем структуру тела запроса
class TextItem(BaseModel):
    text: str

# 4. Функция для предобработки текста (должна совпадать с той, что при обучении)
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 5. Определяем эндпоинт для проверки здоровья сервера
@app.get("/")
def root():
    return {"message": "🚀 Toxicity Detector API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 6. Главный эндпоинт для предсказания
@app.post("/predict/")
async def predict(item: TextItem):
    """
    Принимает текст и возвращает уровень токсичности.
    """
    try:
        # Обрабатываем текст
        cleaned_text = preprocess_text(item.text)

        # Если после очистки текст пуст, возвращаем нейтральный результат
        if not cleaned_text:
            return {"toxicity_score": 0.0, "is_toxic": False, "original_text": item.text}

        # Векторизуем и получаем предсказание
        text_vector = vectorizer.transform([cleaned_text])
        toxicity = model.predict_proba(text_vector)[:, 1][0]

        # ВАЖНО: приводим numpy.bool_ к стандартному bool
        return {
            "toxicity_score": round(float(toxicity), 5),
            "is_toxic": bool(toxicity >= 0.5),
            "original_text": item.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")