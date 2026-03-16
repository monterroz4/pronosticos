from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("pronostico.html")

@app.route('/pronostico', methods=['GET', 'POST'])
def modelo():

    if request.method == "POST":
        archivo = request.files.get('archivo')
        N = int(request.form.get('N'))

        datos = pd.read_csv(archivo)
        productos = datos.columns[1:]  # todas las columnas menos 'periodo'

        resultados = {}

        for producto in productos:
            serie = datos[["periodo", producto]].copy()
            serie.columns = ["periodo", "ventas"]

            serie["pronostico"] = serie["ventas"].rolling(window=N).mean().shift(1)
            serie["error"] = serie["pronostico"] - serie["ventas"]
            serie["error_abs"] = serie["error"].abs()
            serie["ape"] = serie["error_abs"] / serie["ventas"]
            serie["error_cuadrado"] = serie["error"] ** 2

            MAPE = round(serie["ape"].mean(), 4)
            MSE = round(serie["error_cuadrado"].mean(), 4)
            RMSE = round(MSE ** 0.5, 4)

            # pronostico siguiente periodo
            ultimo = serie["ventas"].tail(N).mean()
            proximo = round(ultimo, 2)

            # tabla como lista de diccionarios para el HTML
            tabla = serie.round(2).fillna("").to_dict(orient="records")

            resultados[producto] = {
                "tabla": tabla,
                "MAPE": MAPE,
                "MSE": MSE,
                "RMSE": RMSE,
                "proximo": proximo
            }

        return render_template("pronostico.html", resultados=resultados, N=N)

    return render_template("pronostico.html")


if __name__ == "__main__":
    app.run(debug=True)