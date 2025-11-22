import pandas as pd
import sys
import os

def convert_jsonl_to_csv(input_file, output_file):
    try:
        print(f"🔄 Чтение файла {input_file}...")
        
        # Чтение jsonl файла
        # lines=True указывает, что это именно JSONL (один JSON на строку)
        df = pd.read_json(input_file, lines=True)
        
        # Если нужно оставить только конкретные колонки (например, для LLM), 
        # раскомментируйте строку ниже и укажите нужные поля:
        # df = df[['instruction', 'input', 'output']] 
        
        print(f"📊 Найдено {len(df)} строк. Сохранение в CSV...")
        
        # Сохранение в CSV
        # index=False — не сохранять номер строки
        # escapechar='\\' — помогает избежать ошибок с вложенными кавычками
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Готово! Файл сохранен как: {output_file}")
        
    except ValueError as e:
        print(f"❌ Ошибка формата JSON: {e}")
    except FileNotFoundError:
        print(f"❌ Файл {input_file} не найден.")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    # Использование: python script.py data.jsonl data.csv
    if len(sys.argv) != 3:
        print("Использование: python script.py <входной_файл.jsonl> <выходной_файл.csv>")
    else:
        convert_jsonl_to_csv(sys.argv[1], sys.argv[2])