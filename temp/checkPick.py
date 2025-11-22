import csv
import random
import os
from typing import List, Tuple
from geopy.distance import geodesic

# Центр города — Красная площадь, Москва
CITY_CENTER = (55.7539, 37.6208)  # (широта, долгота)

# Границы генерации координат (вокруг Москвы)
MOSCOW_LAT_CENTER = 55.7558
MOSCOW_LON_CENTER = 37.6173
LAT_SPREAD = 0.3  # ~33 км
LON_SPREAD = 0.4  # ~28 км

def generate_fake_realty_data(n: int = 100) -> List[Tuple[float, float, int, float, float]]:
    """
    Генерирует фиктивные данные по помещениям:
    - latitude
    - longitude
    - area_sqm (20–200 м²)
    - monthly_price_rub (500–5000 ₽/м² × площадь)
    - distance_to_center_km
    """
    data = []
    for _ in range(n):
        lat = round(MOSCOW_LAT_CENTER + random.uniform(-LAT_SPREAD, LAT_SPREAD), 6)
        lon = round(MOSCOW_LON_CENTER + random.uniform(-LON_SPREAD, LON_SPREAD), 6)
        area = random.randint(20, 200)
        price_per_sqm = random.randint(500, 5000)
        total_price = round(area * price_per_sqm, 2)
        
        # Расстояние до центра (в километрах)
        distance_km = round(geodesic(CITY_CENTER, (lat, lon)).km, 2)
        
        data.append((lat, lon, area, total_price, distance_km))
    return data

def save_to_csv(data: List[Tuple], filename: str = "realty_picks.csv"):
    """Сохраняет данные в CSV файл."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "latitude",
            "longitude",
            "area_sqm",
            "monthly_price_rub",
            "distance_to_center_km"
        ])
        writer.writerows(data)
    print(f"Данные сохранены в {os.path.abspath(filename)}")

if __name__ == "__main__":
    # Убедимся, что geopy установлен
    try:
        from geopy.distance import geodesic
    except ImportError:
        print("Библиотека 'geopy' не установлена. Установите её командой:")
        print("   pip install geopy")
        exit(1)

    # Генерируем 100 записей
    fake_data = generate_fake_realty_data(n=100)
    save_to_csv(fake_data)