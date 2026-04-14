import pandas as pd

# Загружаем исходный файл
df = pd.read_csv('result.csv')

# Оставляем только сообщения (type == 'message')
df_messages = df[df['type'] == 'message'].copy()

# Выбираем нужные колонки и переименовываем
df_clean = df_messages[['from', 'text', 'date']].rename(
    columns={'from': 'author', 'text': 'text', 'date': 'date'}
)

# Удаляем строки с пустым текстом (если есть)
df_clean = df_clean.dropna(subset=['text'])

# Преобразуем дату к формату, который pandas сможет прочитать (если нужно)
# В исходном файле дата уже в ISO формате, pandas поймёт её.

# Сохраняем результат
df_clean.to_csv('telegram_clean.csv', index=False, encoding='utf-8')

print(f"Преобразовано сообщений: {len(df_clean)}")
print("Файл сохранён как telegram_clean.csv")