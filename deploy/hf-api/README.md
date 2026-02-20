---
title: Getaround Pricing API
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Getaround Pricing API

API for predicting optimal rental prices for cars.

## Endpoints

- `POST /predict` - Predict rental prices
- `GET /health` - Health check
- `GET /docs` - API documentation (Swagger UI)

## Usage

```bash
curl -X POST https://sam-bot-get-around-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"cars": [{"model_key": "Peugeot", "mileage": 50000, "engine_power": 120, "fuel": "diesel", "paint_color": "black", "car_type": "sedan", "private_parking_available": true, "has_gps": true, "has_air_conditioning": true, "automatic_car": false, "has_getaround_connect": false, "has_speed_regulator": true, "winter_tires": false}]}'
```

## Deployment

This Space is deployed from the Getaround project repository.
