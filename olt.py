# ==============================================
# 1. Importar librerías
# ==============================================
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import pipeline

# ==============================================
# 2. Cargar datos
# ==============================================
df = pd.read_csv("dataset_comunidades_senasoft.csv")

# ==============================================
# 3. Crear figuras básicas
# ==============================================
fig_categorias = px.bar(df, x="Categoría del problema", color="Nivel de urgencia",
                        title="Distribución de problemas por categoría y urgencia")

fig_ciudad = px.histogram(df, x="Ciudad", color="Categoría del problema",
                          title="Distribución de problemas por ciudad")

fig_tiempo = px.histogram(df, x="Fecha del reporte", color="Categoría del problema",
                          title="Frecuencia de reportes en el tiempo")

# ==============================================
# 4. Nube de palabras
# ==============================================
texto = " ".join(df["Comentario"].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texto)

buffer = BytesIO()
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(buffer, format="png")
plt.close()
buffer.seek(0)
wordcloud_base64 = base64.b64encode(buffer.getvalue()).decode()

# ==============================================
# 5. IA generativa (chat)
# ==============================================
modelo_qa = pipeline("text-generation", model="openai-community/gpt2")  # o IBM Granite si lo integras
# Nota: en entornos sin GPU puede usar un modelo más pequeño o conectarse vía API a GPT o Granite.

def responder_pregunta(pregunta):
    respuesta = modelo_qa(pregunta, max_new_tokens=60)[0]["generated_text"]
    return respuesta

# ==============================================
# 6. Construir la app Dash
# ==============================================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Ciudadano con IA", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Análisis general', children=[
            dcc.Graph(figure=fig_categorias),
            dcc.Graph(figure=fig_ciudad),
            dcc.Graph(figure=fig_tiempo),
            html.Img(src=f"data:image/png;base64,{wordcloud_base64}", style={'width': '80%', 'display': 'block', 'margin': 'auto'})
        ]),
        dcc.Tab(label='Chat con IA', children=[
            html.Div([
                html.H3("Hazle preguntas a la IA sobre los datos"),
                dcc.Textarea(id='pregunta', style={'width': '100%', 'height': 100}),
                html.Button('Enviar', id='btn_enviar', n_clicks=0),
                html.Div(id='respuesta', style={'marginTop': 20, 'whiteSpace': 'pre-line'})
            ])
        ])
    ])
])

@app.callback(
    Output('respuesta', 'children'),
    Input('btn_enviar', 'n_clicks'),
    State('pregunta', 'value')
)
def actualizar_respuesta(n_clicks, pregunta):
    if n_clicks > 0 and pregunta:
        return responder_pregunta(pregunta)
    return "Escribe una pregunta y presiona Enviar."

# ==============================================
# 7. Ejecutar servidor
# ==============================================
if __name__ == '__main__':
    app.run_server(debug=True)
