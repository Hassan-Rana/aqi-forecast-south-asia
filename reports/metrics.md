# Evaluation

Test period: 2025-11-22 → 2026-11-22 (trained only on earlier data)

### 1-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 21.8 | 38.7 | 73% | 24.5 |
| Model without occasion features | 21.6 | 38.5 | 73% | 26.2 |
| Persistence (today's AQI) | 23.6 | 43.0 | 70% | 24.2 |
| 7-day mean | 33.1 | 57.5 | 62% | 39.2 |

Per city (model, MAE): delhi 57.0, dhaka 11.6, karachi 9.4, kolkata 22.2, lahore 15.0, mumbai 15.4

### 2-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 28.8 | 49.3 | 64% | 40.4 |
| Model without occasion features | 28.4 | 49.1 | 65% | 37.3 |
| Persistence (today's AQI) | 34.5 | 59.9 | 61% | 39.8 |
| 7-day mean | 36.2 | 62.8 | 60% | 42.8 |

Per city (model, MAE): delhi 69.8, dhaka 15.0, karachi 11.6, kolkata 34.3, lahore 19.1, mumbai 23.0

### 3-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 31.3 | 52.6 | 60% | 44.9 |
| Model without occasion features | 30.8 | 51.7 | 61% | 43.9 |
| Persistence (today's AQI) | 39.8 | 68.0 | 58% | 48.5 |
| 7-day mean | 37.9 | 65.6 | 58% | 44.3 |

Per city (model, MAE): delhi 71.9, dhaka 16.7, karachi 12.0, kolkata 41.5, lahore 20.3, mumbai 25.2
