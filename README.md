<div align="center">

  # Clout or Cancel
<br/>

![Python](https://img.shields.io/badge/Python-%233776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-%23009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5?style=for-the-badge&logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%234DABCF?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458?style=for-the-badge&logo=pandas&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-%23222222?style=for-the-badge&logoColor=white)

</div>

---

X.com sentiment classifier. Given any tweet text, returns a positive or negative verdict with a confidence score, clout score, and cancel risk percentage.

Trained on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset using Logistic Regression over TF-IDF features combined with six hand-crafted signals: text length, word count, uppercase ratio, URL presence, and positive/negative emoji patterns.

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Model info and feature list |
| POST | `/analyse` | Classify a single tweet |
| POST | `/batch` | Classify up to 100 tweets |

## Usage

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"tweet": "just dropped the best update of the year"}'
```

```json
{
  "label": "positive",
  "verdict": "Clout",
  "confidence": 93.7,
  "clout_score": 93.7,
  "cancel_risk": 6.3
}
```

