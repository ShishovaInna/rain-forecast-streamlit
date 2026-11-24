## Streamlit Demo

Цей проєкт розгортає модель машинного навчання, навчену на моделі RandomForestClassifier, з використанням бібліотеки Streamlit. Модель буде використовуватися для прогнозування ймовірності того, що завтра піде дощ, на основі введених користувачем даних про погоду.

Використовуємо набір даних про погоду в Австралії, який містить приблизно 10 років щоденних спостережень за погодою з численних австралійських метеостанцій з файла [weatherAUS.csv](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

## Структура проєкту

- `rain_forest_artifact.joblib`: навчена ML-модель.
- `app.py`: Основний файл застосунку Streamlit.
- `requirements.txt`: Список необхідних Python-пакетів.
- `train_model.ipynb`: Jupyter Notebook для навчання моделі Random Forest.

## Як запустити локально

1. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
2. Запустіть застосунок:
   streamlit run app.py