# Evaluation

Test period: 2025-11-21 → 2026-11-21 (trained only on earlier data)

### 1-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 21.7 | 39.0 | 73% | 26.4 |
| Model without occasion features | 21.4 | 38.5 | 73% | 25.2 |
| Persistence (today's AQI) | 23.7 | 43.1 | 70% | 24.2 |
| 7-day mean | 33.0 | 57.3 | 63% | 39.2 |

Per city (model, MAE): delhi 55.8, dhaka 11.5, karachi 9.5, kolkata 23.0, lahore 14.6, mumbai 15.4

### 2-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 29.5 | 51.3 | 63% | 40.6 |
| Model without occasion features | 28.8 | 49.8 | 64% | 38.9 |
| Persistence (today's AQI) | 34.5 | 59.9 | 61% | 39.8 |
| 7-day mean | 36.1 | 62.5 | 60% | 42.8 |

Per city (model, MAE): delhi 71.0, dhaka 15.0, karachi 11.4, kolkata 37.5, lahore 18.8, mumbai 23.1

### 3-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 32.0 | 54.4 | 61% | 44.2 |
| Model without occasion features | 31.0 | 52.0 | 61% | 44.0 |
| Persistence (today's AQI) | 39.7 | 67.9 | 58% | 48.5 |
| 7-day mean | 37.8 | 65.3 | 58% | 44.3 |

Per city (model, MAE): delhi 74.7, dhaka 16.6, karachi 12.1, kolkata 43.0, lahore 20.3, mumbai 25.2
