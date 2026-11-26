# Evaluation

Test period: 2025-11-20 → 2026-11-20 (trained only on earlier data)

### 1-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 21.7 | 38.6 | 73% | 26.0 |
| Model without occasion features | 21.4 | 38.0 | 73% | 24.7 |
| Persistence (today's AQI) | 23.7 | 43.1 | 70% | 24.2 |
| 7-day mean | 32.8 | 56.9 | 63% | 39.2 |

Per city (model, MAE): delhi 55.5, dhaka 11.6, karachi 9.3, kolkata 23.6, lahore 14.3, mumbai 15.6

### 2-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 29.3 | 50.3 | 63% | 40.2 |
| Model without occasion features | 28.4 | 48.5 | 64% | 38.4 |
| Persistence (today's AQI) | 34.5 | 60.0 | 61% | 39.8 |
| 7-day mean | 35.9 | 62.2 | 60% | 42.8 |

Per city (model, MAE): delhi 70.6, dhaka 15.2, karachi 11.4, kolkata 36.4, lahore 19.2, mumbai 22.9

### 3-day-ahead AQI

| Method | MAE | RMSE | Category accuracy | MAE on festival/event days |
|---|---|---|---|---|
| Model (history + weather + occasions) | 31.6 | 53.5 | 61% | 41.9 |
| Model without occasion features | 31.1 | 52.7 | 62% | 42.4 |
| Persistence (today's AQI) | 39.7 | 67.8 | 58% | 48.5 |
| 7-day mean | 37.7 | 65.1 | 58% | 44.3 |

Per city (model, MAE): delhi 74.1, dhaka 16.6, karachi 11.8, kolkata 41.9, lahore 20.0, mumbai 25.1
