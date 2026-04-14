import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ---------- Настройка страницы ----------
st.set_page_config(page_title="Детектор токсичности команды", layout="wide")
st.title("🔍 Анализ токсичности переписки")
st.markdown("Загрузите файл с перепиской (CSV, JSON экспорт Telegram, Reddit CSV)")

# ---------- Загрузка модели ----------
@st.cache_resource
def load_model():
    with open('baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ---------- Функция предобработки ----------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Парсеры Telegram ----------
def parse_telegram_json(file):
    """Парсит JSON экспорт Telegram, возвращает DataFrame с колонками author, text, date."""
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        st.error(f"❌ Ошибка при чтении JSON файла: {e}. Убедитесь, что вы выбрали правильный тип файла (JSON) и файл не повреждён.")
        return pd.DataFrame()

    messages = []
    for msg in data.get('messages', []):
        if msg.get('type') == 'message':
            author = msg.get('from', 'Unknown')
            text = msg.get('text', '')
            if isinstance(text, list):
                text_parts = []
                for part in text:
                    if isinstance(part, dict):
                        text_parts.append(part.get('text', ''))
                    else:
                        text_parts.append(str(part))
                text = ' '.join(text_parts)
            else:
                text = str(text)
            date = msg.get('date', '')
            if text.strip():
                messages.append([author, text, date])

    df = pd.DataFrame(messages, columns=['author', 'text', 'date'])
    if df.empty:
        st.warning("В JSON не найдено текстовых сообщений. Возможно, это не экспорт Telegram или вы выбрали не тот тип.")
    return df

def parse_telegram_csv(file):
    """Парсит CSV экспорт Telegram (со множеством колонок), возвращает DataFrame с нужными колонками."""
    df_raw = pd.read_csv(file)
    if 'from' in df_raw.columns and 'text' in df_raw.columns and 'date' in df_raw.columns:
        df = df_raw[df_raw['type'] == 'message'][['from', 'text', 'date']].copy()
        df.rename(columns={'from': 'author'}, inplace=True)
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']
        return df
    else:
        st.error("CSV не похож на экспорт Telegram (нет колонок from, text, date). Попробуйте другой тип файла.")
        return None

# ---------- Боковая панель для загрузки ----------
with st.sidebar:
    st.header("📁 Загрузка данных")

    file_type = st.radio(
        "Что вы загружаете?",
        ("Готовый CSV (author,text,date)",
         "Экспорт Telegram (JSON)",
         "Экспорт Telegram (CSV)",
         "Reddit CSV (комментарии)")   # новый тип
    )

    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=['csv', 'json'] if file_type != "Готовый CSV (author,text,date)" else ['csv']
    )

    st.markdown("---")
    st.markdown("### Пример формата")
    if file_type == "Готовый CSV (author,text,date)":
        st.code("""
author,text,date
Иван,Привет всем!,2025-03-01 10:00:00
Пётр,Это бред,2025-03-01 10:01:00
        """)
    elif file_type == "Экспорт Telegram (JSON)":
        st.code("Экспорт из Telegram в формате JSON (обычно result.json)")
    elif file_type == "Экспорт Telegram (CSV)":
        st.code("Экспорт из Telegram в формате CSV (обычно result.csv с множеством колонок)")
    else:  # Reddit CSV (комментарии)
        st.code("""
CSV из Reddit-парсера (колонки: author, body, created_utc, ...)
        """)

    if uploaded_file:
        st.success(f"Файл загружен: {uploaded_file.name}")

# ---------- Основная логика ----------
if uploaded_file is not None:
    try:
        # --- Преобразование в зависимости от типа ---
        if file_type == "Готовый CSV (author,text,date)":
            df = pd.read_csv(uploaded_file)
            required = ['author', 'text', 'date']
            if not all(col in df.columns for col in required):
                st.error(f"Файл должен содержать колонки: {', '.join(required)}")
                st.stop()

        elif file_type == "Экспорт Telegram (JSON)":
            with st.spinner("Парсим JSON экспорт..."):
                df = parse_telegram_json(uploaded_file)
                if df.empty:
                    st.error("Не найдено текстовых сообщений в JSON.")
                    st.stop()
                st.info(f"Загружено сообщений: {len(df)}")

        elif file_type == "Экспорт Telegram (CSV)":
            with st.spinner("Парсим CSV экспорт..."):
                df = parse_telegram_csv(uploaded_file)
                if df is None or df.empty:
                    st.error("Не удалось обработать CSV.")
                    st.stop()
                st.info(f"Загружено сообщений: {len(df)}")

        elif file_type == "Reddit CSV (комментарии)":
            df = pd.read_csv(uploaded_file)
            if 'body' not in df.columns or 'author' not in df.columns or 'created_utc' not in df.columns:
                st.error("Файл должен содержать колонки: author, body, created_utc")
                st.stop()
            df.rename(columns={'body': 'text'}, inplace=True)
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
            df = df[['author', 'text', 'date']]
            df = df.dropna(subset=['text'])
            df = df[df['text'].str.strip() != '']
            st.info(f"Загружено комментариев: {len(df)}")

        else:
            st.error("Неизвестный тип файла")
            st.stop()

        # --- Стандартная обработка ---
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        with st.spinner("Анализ сообщений..."):
            df['clean_text'] = df['text'].apply(preprocess_text)
            X = vectorizer.transform(df['clean_text'])
            df['toxicity'] = model.predict_proba(X)[:, 1]
            df['toxic_binary'] = (df['toxicity'] >= 0.5).astype(int)

        st.success(f"Обработано сообщений: {len(df)}")

        # ---------- Вкладки с результатами ----------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Общая статистика",
            "👥 По участникам",
            "📖 Словарь сообщений",
            "📈 Динамика"
        ])

        with tab1:
            st.subheader("Общие метрики")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Всего сообщений", len(df))
            with col2:
                toxic_pct = df['toxic_binary'].mean() * 100
                st.metric("Токсичных сообщений", f"{toxic_pct:.1f}%")
            with col3:
                st.metric("Участников", df['author'].nunique())
            with col4:
                period = f"{df['date'].min().date()} – {df['date'].max().date()}"
                st.metric("Период", period)

            st.subheader("Распределение токсичности")
            fig, ax = plt.subplots()
            df['toxicity'].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
            ax.axvline(0.5, color='red', linestyle='--', label='Порог 0.5')
            ax.set_xlabel("Уровень токсичности")
            ax.set_ylabel("Количество сообщений")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.subheader("Рейтинг участников")
            author_stats = df.groupby('author').agg({
                'toxicity': 'mean',
                'toxic_binary': 'sum',
                'text': 'count'
            }).rename(columns={
                'toxicity': 'avg_toxicity',
                'toxic_binary': 'toxic_count',
                'text': 'msg_count'
            }).reset_index()
            author_stats['toxic_ratio'] = author_stats['toxic_count'] / author_stats['msg_count']
            author_stats = author_stats.sort_values('avg_toxicity', ascending=False)

            st.dataframe(
                author_stats.style.format({
                    'avg_toxicity': '{:.3f}',
                    'toxic_ratio': '{:.1%}',
                    'toxic_count': '{:.0f}',
                    'msg_count': '{:.0f}'
                }),
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            top10 = author_stats.head(10)
            colors = ['red' if x > 0.5 else 'steelblue' for x in top10['avg_toxicity']]
            ax.barh(top10['author'], top10['avg_toxicity'], color=colors)
            ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
            ax.set_xlabel("Средняя токсичность")
            ax.set_title("Топ-10 участников по токсичности")
            ax.invert_yaxis()
            st.pyplot(fig)

        with tab3:
            st.subheader("Словарь сообщений по участникам")

            authors = ['Все'] + sorted(df['author'].unique().tolist())
            selected_author = st.selectbox("Выберите участника", authors)

            num_words = st.slider("Количество слов для отображения", min_value=5, max_value=50, value=20)

            remove_stopwords = st.checkbox("Убрать служебные и разговорные слова", value=True)

            # Расширенный список стоп-слов
            russian_stopwords = set([
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его',
                'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
                'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
                'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
                'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их',
                'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'человек', 'чего', 'раз', 'тоже', 'себе', 'под', 'жизнь',
                'будет', 'ж', 'тогда', 'кто', 'этот', 'говорил', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
                'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'кажется', 'сейчас', 'были', 'куда',
                'зачем', 'сказать', 'всех', 'никогда', 'сегодня', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
                'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
                'разве', 'сказал', 'эту', 'моя', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя',
                'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'ги', 'это', 'также', 'вроде', 'типа',
                'вообще', 'просто', 'этих', 'этим', 'этими', 'этой', 'этому', 'этого', 'этот',
                # разговорные
                'тебе', 'щас', 'сейчас', 'давай', 'хорошо', 'ок', 'окей', 'ну', 'вот', 'так', 'тоже', 'кстати',
                'вообще', 'просто', 'типа', 'блин', 'короче', 'ладно', 'пока', 'привет', 'здравствуй', 'здравствуйте',
                'спасибо', 'пожалуйста', 'извини', 'извините', 'прости', 'пон', 'понял', 'поняла', 'ага', 'угу',
                'да', 'нет', 'нее', 'неа', 'аа', 'мм', 'ээ', 'хм', 'блин', 'ёбаный', 'блять', 'сука', 'пиздец',
                'хуй', 'хуйня', 'ебать', 'заебал', 'заебала', 'пипец', 'жесть', 'капец', 'конечно', 'наверное',
                'вероятно', 'возможно', 'кажется', 'похоже', 'сегодня', 'завтра', 'вчера', 'утром', 'днём', 'вечером',
                'ночью', 'ща', 'минут', 'час', 'часов', 'дня', 'дней', 'раз', 'два', 'три', 'первый', 'второй',
                'сказал', 'сказала', 'говорил', 'говорила', 'думаю', 'думал', 'думала', 'знаю', 'знаешь', 'понимаю',
                'понимаешь', 'хочешь', 'хочу', 'можешь', 'могу', 'надо', 'нужно', 'можно', 'нельзя', 'обязательно',
                'срочно', 'быстро', 'медленно', 'нормально', 'отлично', 'классно', 'круто', 'супер', 'ужасно', 'плохо',
                'интересно', 'странно', 'глупо', 'смешно', 'прикольно', 'офигенно', 'ахуенно'
            ])

            if selected_author == 'Все':
                texts_series = df['text'].astype(str)
            else:
                texts_series = df[df['author'] == selected_author]['text'].astype(str)

            if not texts_series.empty:
                all_texts = ' '.join(texts_series)
                if all_texts.strip():
                    words = all_texts.split()
                    words = [word.lower() for word in words]
                    if remove_stopwords:
                        words = [word for word in words if word not in russian_stopwords and len(word) > 1]
                    word_counts = pd.Series(words).value_counts().reset_index()
                    word_counts.columns = ['Слово', 'Частота']
                    st.dataframe(
                        word_counts.head(num_words),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Слово": "Слово",
                            "Частота": st.column_config.NumberColumn("Частота", format="%d")
                        }
                    )
                    st.caption(f"Всего уникальных слов: {len(word_counts)}")
                else:
                    st.warning("У выбранного участника нет текстовых сообщений.")
            else:
                st.warning("Нет сообщений для выбранного участника.")

        with tab4:
            st.subheader("Динамика токсичности по дням")
            df['day'] = df['date'].dt.date
            daily = df.groupby('day').agg({
                'toxicity': 'mean',
                'toxic_binary': 'sum',
                'text': 'count'
            }).rename(columns={
                'toxicity': 'avg_toxicity',
                'toxic_binary': 'toxic_count',
                'text': 'msg_count'
            })
            daily['toxic_ratio'] = daily['toxic_count'] / daily['msg_count']

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(daily.index, daily['avg_toxicity'], marker='o', linestyle='-', label='Средняя токсичность')
            ax.axhline(0.5, color='red', linestyle='--', label='Порог')
            ax.set_xlabel("Дата")
            ax.set_ylabel("Токсичность")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            if len(daily) >= 3:
                last_days = daily['avg_toxicity'].tail(3).values
                if last_days[-1] > last_days[0] and last_days[-1] > 0.5:
                    st.warning("⚠️ За последние 3 дня токсичность растёт и превышает порог!")


    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")

else:
    st.info("👈 Загрузите файл в боковой панели")
    st.markdown("""
    ### Как подготовить экспорт из Telegram:
    1. В Telegram Desktop откройте чат
    2. Нажмите на три точки → "Экспорт истории чата"
    3. Выберите формат JSON или CSV
    4. Загрузите полученный файл сюда

    ### Для Reddit:
    Используйте парсер (например, Apify Reddit Comment Scraper), экспортируйте в CSV и загрузите как "Reddit CSV (комментарии)".
    """)