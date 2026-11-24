# Reporte del proyecto: ¿Comprará la casa?

**Objetivo:** predecir si un cliente comprará una vivienda a partir de sus características y las de la propiedad, y publicar una app donde un vendedor capture esos datos y obtenga la probabilidad de compra.

## Entregables

| # | Entregable | Enlace |
|---|---|---|
| 1 | App en línea (Streamlit Community Cloud) | `https://<tu-app>.streamlit.app` |
| 2 | Estadísticas y gráficas de entrenamiento (Weights & Biases) | `https://wandb.ai/<tu-usuario>/house-purchase` |
| 3 | Este reporte | — |

## 1. Datos

- Fuente: *Global House Purchase Decision Dataset* (Kaggle, mohankrishnathalla). 200 000 registros, 25 columnas, sin valores faltantes ni duplicados.
- Objetivo: `decision` (1 = compra). Clases desbalanceadas: **23.0 % compra / 77.0 % no compra**.
- Variables: 4 categóricas (país, ciudad, tipo de propiedad, amueblado) y 19 numéricas (tamaño, precio, año, recámaras, baños, cochera, jardín, delitos en la zona, litigios, ingreso, crédito, plazo, gastos, enganche, mensualidad/ingreso, satisfacción, calificación del vecindario, conectividad). Se descartó `property_id`.
- Correlación con la compra: `satisfaction_score` (+0.57), `legal_cases_on_property` (−0.32), `crime_cases_reported` (−0.17), `emi_to_income_ratio` (−0.16), `customer_salary` (+0.09). Las demás están prácticamente en 0.

## 2. Metodología

- **División estratificada** (semilla 42): **70 % entrenamiento (140 000), 10 % validación (20 000), 20 % prueba (40 000)**.
- **Modelos comparados:** regresión logística (con estandarización y one-hot), árbol de decisión (profundidad 8), random forest (200 árboles) y LightGBM (early stopping de 100 rondas sobre la validación).
- **Selección:** el modelo ganador se elige por ROC-AUC en **validación**; el conjunto de **prueba** solo se usa para reportar.
- **Umbral:** como la clase positiva es minoritaria, además del umbral 0.5 se busca el umbral que maximiza F1 en validación y se reporta su desempeño en prueba.
- **Dos escenarios de uso**, porque no todos los datos están disponibles en el mismo momento de la venta:
  - **Con visita:** incluye la satisfacción del cliente después de ver la casa (el vendedor se la pregunta al terminar la visita).
  - **Sin visita:** mismo modelo sin `satisfaction_score`, para estimar la probabilidad antes de enseñar la casa.
- **Registro de experimentos:** cada uno de los 8 entrenamientos es un *run* de Weights & Biases (proyecto `house-purchase`, agrupado por escenario) con hiperparámetros, métricas de entrenamiento/validación/prueba, curva de aprendizaje de LightGBM, importancia de variables, curva ROC y matriz de confusión.

## 3. Resultados (conjunto de prueba, 40 000 registros)

### Escenario con visita

| Modelo | Accuracy | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística | 0.9431 | 0.8764 | 0.8764 | 0.8764 | 0.9860 |
| **Árbol de decisión (elegido)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Ante el empate se queda el árbol de decisión: es el más simple (2.6 KB), es interpretable y tiene la menor pérdida logarítmica.

### Escenario sin visita

| Modelo | Accuracy (0.5) | F1 (0.5) | F1 (umbral ajustado) | Precisión / Recall (umbral ajustado) | ROC-AUC |
|---|---|---|---|---|---|
| Regresión logística | 0.7607 | 0.1350 | 0.5530 | — | 0.7715 |
| Árbol de decisión | 0.7693 | 0.0065 | 0.5727 | — | 0.7748 |
| Random forest | 0.7697 | 0.0000 | 0.5727 | — | 0.7748 |
| **LightGBM (elegido)** | 0.7690 | 0.0058 | **0.5725** | 0.40 / 1.00 (umbral 0.28) | **0.7765** |

Sin la satisfacción, el modelo distingue bien a los clientes que **seguro no compran** (litigios, zona insegura o mensualidad impagable) pero no puede saber si a uno viable le gustará la casa. Su probabilidad máxima ronda el **40 %**, que es justo la fracción de clientes que califican la casa con 7 o más. Por eso, con umbral 0.5, casi nunca predice "compra" y el F1 cae a ~0; con el umbral ajustado (0.28) encuentra al 99.9 % de los compradores con una precisión del 40 %. Para el vendedor, la salida útil es la **probabilidad**, no la clase.

### Variables más importantes

| Variable | Con visita (árbol) | Sin visita (LightGBM) |
|---|---|---|
| Satisfacción tras la visita | 45 % | — |
| Litigios sobre la propiedad | 25 % | 47 % |
| Delitos reportados en la zona | 16 % | 26 % |
| Mensualidad / ingreso | 15 % | 22 % |
| Resto (ciudad, precio, tamaño, etc.) | ~0 % | < 5 % en conjunto |

## 4. Problemas que se presentaron

1. **La etiqueta es una regla determinista (datos sintéticos).** Un árbol de profundidad 4 reproduce el 100 % de las etiquetas. La regla exacta es: *compra si y solo si* satisfacción ≥ 7, **y** no hay litigios, **y** hay ≤ 2 delitos reportados, **y** mensualidad/ingreso ≤ 0.39. Por eso el escenario con visita llega a 100 % en prueba. No es sobreajuste (validación y prueba dan lo mismo), pero tampoco es realista: con datos reales de una inmobiliaria la precisión sería mucho menor. Además, la probabilidad que da el modelo con visita es prácticamente 0 % o 100 %.
2. **Posible fuga de información con `satisfaction_score`.** Es la variable más predictiva y solo existe después de la visita. Para no inflar artificialmente el desempeño se entrenó el escenario *sin visita* y la app pregunta si el cliente ya vio la casa.
3. **Desbalance de clases (23 % / 77 %).** Con umbral 0.5, el modelo sin visita tiene buena *accuracy* (0.77) y F1 casi nulo: predecir siempre "no compra" ya da 77 %. Se resolvió ajustando el umbral en validación y reportando ROC-AUC y F1, no solo accuracy.
4. **Early stopping de LightGBM detenido en la iteración 1.** LightGBM evaluaba primero el AUC, que llega a 1.0 desde el inicio y deja de mejorar, así que el entrenamiento se cortaba en la iteración 1 con probabilidades ~0.23 (accuracy 0.77, F1 0). Se corrigió fijando la *log loss* como métrica principal (`first_metric_only=True`).
5. **Rutas largas en Windows.** W&B no pudo crear su carpeta temporal (`WinError 206`, ruta > 260 caracteres). Se resolvió con `WANDB_DIR` apuntando a una ruta corta.
6. **Variables derivadas.** `loan_amount` es exactamente `price − down_payment`, y `emi_to_income_ratio` se reconstruyó como la mensualidad de un crédito al **6.5 % anual** entre el ingreso mensual (coincide en el 100 % de los registros). Así la app las calcula sola y el vendedor no tiene que capturarlas.
7. **Consistencia del formulario.** En Streamlit, los campos dentro de un `st.form` no se actualizan hasta enviar, así que la lista de ciudades no cambiaba al elegir otro país. País y ciudad se sacaron del formulario.

## 5. La app

Formulario con los datos que un vendedor puede obtener:

- **Propiedad:** país, ciudad, tipo, amueblado, tamaño, precio, año, dueños anteriores, recámaras, baños, cochera, jardín.
- **Zona:** delitos reportados (dato público), litigios sobre la propiedad, calificación del vecindario, conectividad.
- **Cliente:** ingreso y gastos mensuales, enganche, plazo del crédito y, si ya visitó la casa, qué tanto le gustó (1–10).

La app calcula el crédito, la mensualidad y la razón mensualidad/ingreso. Elige el modelo *con visita* o *sin visita* según el caso y muestra la probabilidad, un semáforo de los factores que más pesan (con sugerencias, por ejemplo "pide más enganche o más plazo") y una pestaña para evaluar varios clientes desde un CSV.

## 6. Costo estimado por mes

La app es ligera: modelos de 2.6 KB y 208 KB, menos de 300 MB de RAM, predicción en milisegundos y sin GPU.

| Opción | Costo mensual aprox. (USD) | Comentario |
|---|---|---|
| **Streamlit Community Cloud + W&B plan gratuito (la configuración usada)** | **$0** | La app se duerme tras unos días sin visitas y tarda ~30 s en despertar. Solo repos públicos. |
| Hugging Face Spaces (CPU básico) | $0 | Alternativa gratuita equivalente. |
| Google Cloud Run (escala a cero) | $0 – $5 | Con poco tráfico cae dentro de la capa gratuita; se paga por uso. |
| Render / Railway (instancia pequeña siempre encendida) | $5 – $7 | Sin "dormirse", dominio propio. |
| VM pequeña (AWS t3.small / GCP e2-small) | $13 – $17 | Control total, mantenimiento propio. |
| Dominio propio (opcional) | ~$1 | ~$12 al año. |
| W&B plan de pago (solo si hay equipo de varias personas) | ~$50 por usuario | No hace falta para este proyecto. |

**Estimación del proyecto en producción ligera:** **$0/mes** tal como está, o **~$7–8/mes** si se quiere una app siempre encendida con dominio propio. Precios de referencia de noviembre de 2026; verificar en cada proveedor.

## 7. Conclusiones

- Con todos los datos (incluida la satisfacción tras la visita), el problema se resuelve perfectamente porque el dataset sigue una regla fija. El valor práctico está en **explicar al vendedor qué factor frena la venta**.
- Antes de la visita, el modelo filtra con confianza a los clientes que no comprarán (AUC 0.78) y le dice al vendedor dónde enfocar su tiempo.
- Siguiente paso recomendado: validar con datos reales de operaciones, donde la relación no será determinista, y recalibrar las probabilidades.
