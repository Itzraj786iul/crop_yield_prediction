# Crop Yield Prediction API

This API provides predictions for crop yields based on historical data and various features.

## Features

- Predict crop yield using a trained Random Forest model
- Web interface for easy interaction
- RESTful API endpoints

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`

## Usage

- Web Interface: Open http://localhost:5000 in your browser
- API Endpoint: Send POST requests to http://localhost:5000/predict

## API Documentation

### POST /predict

Predict crop yield based on input features.

**Request Body:**
```json
{
  "Year": 2018,
  "State Code": 14,
  "Dist Code": 1,
  "State Name": "Chhattisgarh",
  "Dist Name": "Durg",
  "crop": "RICE",
  "AREA": 100.0,
  "district_mean_yield": 1500.0,
  "state_mean_yield": 1600.0,
  "prev_year_yield": 1400.0
}