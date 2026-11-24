# ¿Comprará la casa? — Predicción de compra de vivienda

Clasificador binario sobre el [Global House Purchase Decision Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/global-house-purchase-decision-dataset)
(200 000 registros) y app web en Streamlit para vendedores.

| Archivo | Qué es |
|---|---|
| `train.py` | Split 70/10/20, entrena 4 modelos × 2 escenarios y registra todo en Weights & Biases |
| `app.py` | App Streamlit (formulario de un cliente + carga CSV de varios) |
| `models/` | Modelos ganadores (`con_visita.joblib`, `sin_visita.joblib`) y `metadata.json` |
| `reports/` | Métricas (`metrics.json`) y gráficas ROC / matriz de confusión |
| `REPORTE.md` | Reporte del proyecto |

## Correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reentrenar y registrar en Weights & Biases

```bash
pip install -r requirements-train.txt
wandb login                      # pega tu API key de https://wandb.ai/authorize
python train.py --data global_house_purchase_dataset.csv --project house-purchase
```

## Publicar en Streamlit Community Cloud (gratis)

1. Sube esta carpeta a un repositorio público de GitHub.
2. En https://share.streamlit.io → **Create app** → elige el repo, rama `main`, archivo `app.py`.
3. En **Advanced settings** elige Python 3.13 y despliega.
