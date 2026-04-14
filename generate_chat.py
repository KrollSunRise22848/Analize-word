import csv
import random
from datetime import datetime, timedelta

# Настройки
NUM_MESSAGES = 100          # количество сообщений
AUTHORS = ['Анна', 'Пётр', 'Елена', 'Модератор', 'Иван', 'Ольга']
START_DATE = datetime(2025, 3, 1, 9, 0, 0)
END_DATE = datetime(2025, 3, 5, 18, 0, 0)
TOXIC_PROB = 0.3            # вероятность токсичного сообщения (для Пети выше)

# Шаблоны сообщений
NEUTRAL_TEXTS = [
    "Привет всем!", "Как дела?", "Что нового?", "Я сделал задачу",
    "Когда созвон?", "Нужна помощь", "Готово", "Ок", "Понял",
    "Давай завтра", "Отлично", "Спасибо", "Хорошего дня"
]
TOXIC_TEXTS = [
    "Ты тупой?", "Это полный бред", "Заткнись", "Иди ты",
    "Ничего не понимаешь", "Руки из жопы", "Дурак", "Бесишь",
    "Отвали", "Какой же ты дебил", "Сам такой", "Мозгов нет"
]

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

messages = []
for i in range(NUM_MESSAGES):
    author = random.choice(AUTHORS)
    # Повысим токсичность для Пети
    prob = TOXIC_PROB
    if author == 'Пётр':
        prob = min(0.8, prob * 2)  # Пётр токсичнее
    is_toxic = random.random() < prob
    if is_toxic:
        text = random.choice(TOXIC_TEXTS)
    else:
        text = random.choice(NEUTRAL_TEXTS)
    date = random_date(START_DATE, END_DATE)
    messages.append([author, text, date.strftime('%Y-%m-%d %H:%M:%S')])

# Сортируем по дате
messages.sort(key=lambda x: x[2])

with open('generated_chat.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['author', 'text', 'date'])
    writer.writerows(messages)

print(f"Сгенерировано {NUM_MESSAGES} сообщений в файл generated_chat.csv")