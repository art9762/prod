import pandas as pd

# Укажите имя вашего файла
input_file = '/Users/artem/Documents/GitHub/ai_training/datasets/cian_data.csv'
output_file = '/Users/artem/Documents/GitHub/ai_training/datasets/cian_data_cleaned.csv'

# Читаем CSV файл
df = pd.read_csv(input_file)

# Описание условий для УДАЛЕНИЯ:
# 1. sq_price_amt > 70000000
# 2. floor_no НЕ равен 0 и НЕ равен 1
condition_to_remove = (
    (df['sq_price_amt'] > 50000000) | 
    (~df['floor_no'].isin([0, 1]))
)

# Оставляем только те строки, которые НЕ подходят под условие удаления
# Знак ~ означает "НЕ" (инверсия)
df_filtered = df[~condition_to_remove]

# Сохраняем результат в новый файл
df_filtered.to_csv(output_file, index=False)

print(f"Готово! Удалено строк: {len(df) - len(df_filtered)}")