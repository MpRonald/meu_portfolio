# services.py
import requests


class PortfolioService:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # -------- Weather (Open-Meteo) --------
    def geocode_city(self, name: str, count: int = 1, lang: str = "pt") -> dict:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": name, "count": count, "language": lang, "format": "json"}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        res = (js.get("results") or [])
        if not res:
            return {}
        top = res[0]
        return {
            "name": top.get("name"),
            "country": top.get("country"),
            "latitude": float(top["latitude"]),
            "longitude": float(top["longitude"]),
            "timezone": top.get("timezone"),
        }

    def fetch_weather_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        temp_unit: str = "celsius",
        wind_unit: str = "kmh",
        lang: str = "pt",
    ) -> dict:
        url = "https://api.open-meteo.com/v1/forecast"
        daily = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(daily),
            "forecast_days": max(1, min(30, int(days))),
            "timezone": "auto",
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "past_days": 0,
            "language": lang,
        }
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
