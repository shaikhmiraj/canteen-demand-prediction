# AI-Based College Canteen Food Demand Prediction System

## Setup and run (Windows / Linux / Mac)
```
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python generate_data.py        # creates data/canteen_sales.csv
python train_model.py          # trains model, prints MAE and R2, saves model/
streamlit run app.py           # opens the web app at http://localhost:8501
```
