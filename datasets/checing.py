import csv
import random
import os
from typing import List, Tuple

# Генерация случайных координат в пределах Москвы (примерно)
MOSCOW_LAT_CENTER = 55.7558
MOSCOW_LON_CENTER = 37.6173
LAT_SPREAD = 0.2
LON_SPREAD = 0.3

def generate_fake_realty_data(n: int = 100) -> List[Tuple[float, float, int, float]]:
    """
    Генерирует фиктивные данные по помещениям:
    - координаты (широта, долгота)
    - площадь (от 20 до 200 м²)
    - стоимость аренды в месяц (руб., от 500 до 5000 руб/м²)
    """
    data = []
    for _ in range(n):
        lat = round(MOSCOW_LAT_CENTER + random.uniform(-LAT_SPREAD, LAT_SPREAD), 6)
        lon = round(MOSCOW_LON_CENTER + random.uniform(-LON_SPREAD, LON_SPREAD), 6)
        area = random.randint(20, 200)
        price_per_sqm = random.randint(500, 5000)
        total_price = round(area * price_per_sqm, 2)
        data.append((lat, lon, area, total_price))
    return data

def save_to_csv(data: List[Tuple], filename: str = "realty_picks.csv"):
    """Сохраняет данные в CSV файл."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["latitude", "longitude", "area_sqm", "monthly_price_rub"])
        writer.writerows(data)
    print(f"Данные сохранены в {os.path.abspath(filename)}")

if __name__ == "__main__":
    # Генерируем 100 записей
    fake_data = generate_fake_realty_data(n=100)
    save_to_csv(fake_data)