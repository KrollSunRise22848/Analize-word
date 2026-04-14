import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Скачиваем стоп-слова для русского языка (делаем один раз)
nltk.download('stopwords')

# Загружаем данные (предполагаем, что файл labeled.csv в текущей папке)
df = pd.read_csv('labeled.csv')

# Проверим, какие колонки есть в файле
print("Колонки в файле:", df.columns.tolist())
print("Первые 5 строк:")
print(df.head())

# Предположим, что текст в колонке 'comment', а метка в 'toxic'
# Если названия отличаются, подставьте свои
text_column = 'comment'   # измените, если нужно
label_column = 'toxic'    # измените, если нужно

# Убедимся, что метки целочисленные
df[label_column] = df[label_column].astype(int)

# Проверим распределение классов
print("\nРаспределение классов:")
print(df[label_column].value_counts())

# Функция очистки текста
def preprocess_text(text):
    # Приводим к нижнему регистру
    text = str(text).lower()
    # Удаляем всё кроме букв и пробелов
    text = re.sub(r'[^а-яё\s]', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Токенизация и удаление стоп-слов
    tokens = text.split()
    # Загружаем стоп-слова (они уже скачаны)
    russian_stopwords = set(stopwords.words('russian'))
    tokens = [token for token in tokens if token not in russian_stopwords]
    # Возвращаем строку
    return ' '.join(tokens)

# Применяем очистку ко всем текстам (может занять некоторое время)
print("\nОчищаем тексты...")
df['clean_text'] = df[text_column].astype(str).apply(preprocess_text)

# Разделяем на признаки и целевую переменную
X = df['clean_text']
y = df[label_column]

# Разбиваем на обучающую и тестовую выборки (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Векторизация TF-IDF
tfidf = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 1))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Размер обучающей матрицы: {X_train_tfidf.shape}")
print(f"Размер тестовой матрицы: {X_test_tfidf.shape}")

# Обучение модели
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Предсказания и оценка
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f}")
print("\nОтчёт по метрикам:")
print(classification_report(y_test, y_pred, target_names=['Нейтральный', 'Токсичный']))

# Сохраняем модель и векторизатор
with open('baseline_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nМодель и векторизатор сохранены в файлы.")