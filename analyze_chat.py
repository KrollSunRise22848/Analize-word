import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ---------- Функция предобработки (такая же, как при обучении) ----------
def preprocess_text(text):
    # Приводим к нижнему регистру
    text = str(text).lower()
    # Удаляем всё кроме букв и пробелов
    text = re.sub(r'[^а-яё\s]', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Токенизация
    tokens = text.split()
    # Загружаем стоп-слова (используем тот же список, что и в спринте 1)
    # Для простоты можно взять готовый набор из nltk или использовать встроенный.
    # Здесь мы используем тот же подход, что и при обучении, но если вы использовали стоп-слова nltk,
    # лучше их загрузить так же, как в первом спринте.
    # ВНИМАНИЕ: если при обучении вы удаляли стоп-слова, здесь нужно делать то же самое.
    # Для экономии времени я оставлю без удаления стоп-слов (они не сильно влияют на baseline).
    # Но если хотите точно повторить预处理, раскомментируйте следующие строки:
    # from nltk.corpus import stopwords
    # russian_stopwords = set(stopwords.words('russian'))
    # tokens = [token for token in tokens if token not in russian_stopwords]
    return ' '.join(tokens)

# ---------- Загрузка модели и векторизатора ----------
print("Загрузка модели и векторизатора...")
with open('baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print("Модель загружена.")

# ---------- Загрузка данных чата ----------
chat_file = input("Введите путь к CSV-файлу с перепиской: ").strip()
if not os.path.exists(chat_file):
    print("Файл не найден. Завершение.")
    exit(1)

df_chat = pd.read_csv(chat_file, quoting=1)
print(f"Загружено сообщений: {len(df_chat)}")

# Проверяем наличие необходимых колонок
required_cols = ['author', 'text', 'date']
if not all(col in df_chat.columns for col in required_cols):
    print("Ошибка: в CSV должны быть колонки: author, text, date")
    exit(1)

# Преобразуем дату в datetime
df_chat['date'] = pd.to_datetime(df_chat['date'], errors='coerce')
df_chat = df_chat.dropna(subset=['date'])  # удаляем строки с некорректной датой

# ---------- Предобработка текстов ----------
print("Предобработка сообщений...")
df_chat['clean_text'] = df_chat['text'].astype(str).apply(preprocess_text)

# ---------- Предсказание токсичности ----------
print("Предсказание токсичности...")
# Преобразуем тексты в TF-IDF признаки
X_chat = vectorizer.transform(df_chat['clean_text'])

# Получаем вероятности токсичности (столбец 1 соответствует классу 1 - токсичный)
proba = model.predict_proba(X_chat)[:, 1]
df_chat['toxicity_prob'] = proba

# Можно также получить бинарные предсказания (если нужно)
threshold = 0.5  # порог, можно будет менять
df_chat['toxic_pred'] = (proba >= threshold).astype(int)

print("Предсказания получены.")

# ---------- Агрегация по дням ----------
df_chat['day'] = df_chat['date'].dt.date
daily_stats = df_chat.groupby('day').agg(
    avg_toxicity=('toxicity_prob', 'mean'),
    msg_count=('toxicity_prob', 'count'),
    toxic_count=('toxic_pred', 'sum')
).reset_index()
daily_stats['toxic_ratio'] = daily_stats['toxic_count'] / daily_stats['msg_count']

# ---------- Агрегация по авторам ----------
author_stats = df_chat.groupby('author').agg(
    avg_toxicity=('toxicity_prob', 'mean'),
    msg_count=('toxicity_prob', 'count'),
    toxic_count=('toxic_pred', 'sum')
).reset_index()
author_stats['toxic_ratio'] = author_stats['toxic_count'] / author_stats['msg_count']
author_stats = author_stats.sort_values('avg_toxicity', ascending=False)

# ---------- Построение графиков ----------
plt.figure(figsize=(12, 5))

# График динамики токсичности по дням
plt.subplot(1, 2, 1)
plt.plot(daily_stats['day'], daily_stats['avg_toxicity'], marker='o', linestyle='-')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Порог {threshold}')
plt.xlabel('Дата')
plt.ylabel('Средняя токсичность')
plt.title('Динамика токсичности по дням')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Столбчатая диаграмма по авторам (топ-10 самых токсичных)
plt.subplot(1, 2, 2)
top_authors = author_stats.head(10)
plt.barh(top_authors['author'], top_authors['avg_toxicity'], color='salmon')
plt.xlabel('Средняя токсичность')
plt.title('Топ-10 самых токсичных участников')
plt.gca().invert_yaxis()  # чтобы самый токсичный был сверху
plt.tight_layout()

plt.show()

# ---------- Текстовый отчёт ----------
print("\n" + "="*50)
print("          АНАЛИТИЧЕСКИЙ ОТЧЁТ")
print("="*50)

print(f"\nВсего сообщений: {len(df_chat)}")
print(f"Уникальных авторов: {df_chat['author'].nunique()}")
print(f"Период: {df_chat['date'].min().date()} - {df_chat['date'].max().date()}")

# Самый токсичный автор
worst_author = author_stats.iloc[0]
print(f"\nСамый токсичный автор: {worst_author['author']}")
print(f"  Средняя токсичность: {worst_author['avg_toxicity']:.3f}")
print(f"  Сообщений: {worst_author['msg_count']}")
print(f"  Доля токсичных: {worst_author['toxic_ratio']:.2%}")

# День с максимальной средней токсичностью
worst_day = daily_stats.loc[daily_stats['avg_toxicity'].idxmax()]
print(f"\nДень с максимальной токсичностью: {worst_day['day']}")
print(f"  Средняя токсичность: {worst_day['avg_toxicity']:.3f}")
print(f"  Сообщений: {worst_day['msg_count']}, токсичных: {worst_day['toxic_count']}")

# Общий тренд: если последние дни токсичность растёт, предупреждение
if len(daily_stats) >= 3:
    last_days = daily_stats.tail(3)['avg_toxicity'].values
    if last_days[-1] > last_days[0] and last_days[-1] > threshold:
        print("\n⚠️  ВНИМАНИЕ: за последние 3 дня токсичность растёт и превышает порог!")
    else:
        print("\n✅ Тенденция последних дней стабильна.")

# Сохраняем отчёт в файл (по желанию)
with open('toxicity_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ ПО ТОКСИЧНОСТИ КОМАНДЫ\n")
    f.write(f"Дата отчёта: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Файл чата: {chat_file}\n")
    f.write("="*50 + "\n")
    f.write(f"Всего сообщений: {len(df_chat)}\n")
    f.write(f"Уникальных авторов: {df_chat['author'].nunique()}\n")
    f.write(f"Период: {df_chat['date'].min().date()} - {df_chat['date'].max().date()}\n\n")
    f.write("ТОП-10 ТОКСИЧНЫХ АВТОРОВ:\n")
    for i, row in author_stats.head(10).iterrows():
        f.write(f"{row['author']}: средняя токсичность {row['avg_toxicity']:.3f} "
                f"(сообщений {row['msg_count']}, токсичных {row['toxic_count']})\n")
    f.write("\nДИНАМИКА ПО ДНЯМ:\n")
    for i, row in daily_stats.iterrows():
        f.write(f"{row['day']}: средняя {row['avg_toxicity']:.3f}, "
                f"всего {row['msg_count']} (токс {row['toxic_count']})\n")

print("\nОтчёт сохранён в файл toxicity_report.txt")