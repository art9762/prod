import json
import csv 
import os
input_file_name = 'cian_data.jsonlines'
output_file_name = 'cian_data.csv'
if not os.path.exists(input_file_name):
    print(f"Ошибка: Файл '{input_file_name}' не найден.")
    print("Пожалуйста, убедитесь, что файл находится в той же папке, что и скрипт, или укажите полный путь.")
else:
    print(f"Начинаю обработку файла: {input_file_name}")
# --- Шаг 1: Чтение всех данных и сбор всех уникальных ключей (заголовков) ---
all_data = []
all_keys = set()
line_number = 0
try:
    with open(input_file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            # Пропускаем пустые строки, если они есть
            if not line.strip():
                continue
            try:
                # Преобразуем строку JSON в словарь Python
                data_dict = json.loads(line)
                all_data.append(data_dict)
                # Добавляем ключи из этого словаря в общий набор ключей
                all_keys.update(data_dict.keys())
            except json.JSONDecodeError:
                print(f"Предупреждение: Не удалось прочитать строку {line_number} как JSON. Строка пропущена.")
except FileNotFoundError:
    # Эта проверка уже есть выше, но на всякий случай
    print(f"Ошибка: Не удалось открыть файл '{input_file_name}'.")
    all_data = [] # Очищаем данные, чтобы скрипт не продолжил работу

# --- Шаг 2: Запись данных в CSV файл ---
if not all_data:
    print("Входной файл пуст или не содержит корректных JSON-объектов. CSV-файл не будет создан.")
else:
    # Преобразуем набор ключей в отсортированный список для постоянного порядка колонок
    headers = sorted(list(all_keys))

    print(f"Обнаружено {len(all_data)} строк и {len(headers)} уникальных колонок.")
    print("Начинаю запись в CSV файл...")

    try:
        with open(output_file_name, 'w', newline='', encoding='utf-8') as f:
            # Используем DictWriter, который работает со словарями
            # fieldnames - это названия колонок (наши заголовки)
            writer = csv.DictWriter(f, fieldnames=headers)

            # Записываем первую строку с заголовками
            writer.writeheader()

            # Записываем все данные из списка словарей
            # DictWriter сам подставит пустые значения, если у словаря нет какого-то ключа
            writer.writerows(all_data)

        print(f"Успешно! Данные сохранены в файл: {output_file_name}")

    except IOError:
        print(f"Ошибка: Не удалось записать данные в файл '{output_file_name}'.")