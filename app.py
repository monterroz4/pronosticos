from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---- función de errores ----
def calcular_errores(real, pronostico):
    real = pd.Series(real).reset_index(drop=True)
    pronostico = pd.Series(pronostico).reset_index(drop=True)
    df = pd.DataFrame({'ventas': real, 'pronostico': pronostico}).dropna()
    if len(df) == 0:
        return "N/A", "N/A", "N/A"
    df['error_abs'] = (df['pronostico'] - df['ventas']).abs()
    df['ape'] = df['error_abs'] / df['ventas']
    df['error_cuadrado'] = (df['pronostico'] - df['ventas']) ** 2
    MAPE = round(df['ape'].mean(), 4)
    MSE  = round(df['error_cuadrado'].mean(), 4)
    RMSE = round(MSE ** 0.5, 4)
    return MAPE, MSE, RMSE

@app.route('/')
def home():
    return render_template("pronostico.html")

@app.route('/pronostico', methods=['GET', 'POST'])
def modelo():
    if request.method == "POST":

        archivo          = request.files.get('archivo')
        N                = int(request.form.get('N') or 3)
        metodo           = request.form.get('metodo') or 'PM'
        periodos_futuros = int(request.form.get('periodos_futuros') or 3)

        datos     = pd.read_csv(archivo)
        productos = list(datos.columns[1:])
        periodos  = datos['periodo'].tolist()

        resultados = {}
        resumen    = {}

        for producto in productos:
            serie = datos[producto].astype(float)

            # =============================
            # MODELO 1: Promedio Movil
            # model -> fit -> predict
            # =============================
            pm_model      = serie.rolling(window=N)
            pm_fitted     = pm_model.mean().shift(1)
            pm_predict    = [round(serie.tail(N).mean(), 2)] * periodos_futuros
            pm_MAPE, pm_MSE, pm_RMSE = calcular_errores(serie, pm_fitted)

            # =============================
            # MODELO 2: Suavizacion Exponencial Simple
            # model -> fit -> predict
            # =============================
            ses_model     = SimpleExpSmoothing(serie.values)
            ses_fit       = ses_model.fit(optimized=True)
            ses_fitted    = pd.Series(ses_fit.fittedvalues)
            ses_predict   = [round(v, 2) for v in ses_fit.forecast(periodos_futuros)]
            ses_MAPE, ses_MSE, ses_RMSE = calcular_errores(serie, ses_fitted)

            # =============================
            # MODELO 3: Prophet (Meta)
            # model -> fit -> predict
            # =============================
            df_prophet = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(serie), freq='MS'),
                'y' : serie.values
            })
            prophet_model = Prophet(yearly_seasonality=False,
                                    weekly_seasonality=False,
                                    daily_seasonality=False)
            prophet_fit   = prophet_model.fit(df_prophet)
            futuro        = prophet_model.make_future_dataframe(periods=periodos_futuros, freq='MS')
            prophet_pred  = prophet_fit.predict(futuro)
            prophet_fitted  = list(prophet_pred['yhat'][:len(serie)].values)
            prophet_predict = [round(v, 2) for v in prophet_pred['yhat'].tail(periodos_futuros).values]
            prophet_MAPE, prophet_MSE, prophet_RMSE = calcular_errores(serie, pd.Series(prophet_fitted))

            # ---- tabla resumen comparativa ----
            resumen[producto] = {
                'Promedio Movil': {'MAPE': pm_MAPE,      'MSE': pm_MSE,      'RMSE': pm_RMSE},
                'SES'           : {'MAPE': ses_MAPE,      'MSE': ses_MSE,     'RMSE': ses_RMSE},
                'Prophet'       : {'MAPE': prophet_MAPE,  'MSE': prophet_MSE, 'RMSE': prophet_RMSE}
            }

            # ---- método seleccionado ----
            if metodo == 'PM':
                fitted_sel  = [round(v, 2) if v == v else "" for v in pm_fitted.tolist()]
                predict_sel = pm_predict
            elif metodo == 'SES':
                fitted_sel  = [round(v, 2) for v in ses_fitted.tolist()]
                predict_sel = ses_predict
            else:
                fitted_sel  = [round(v, 2) for v in prophet_fitted]
                predict_sel = prophet_predict

            # ---- tabla detallada ----
            tabla = []
            for i, p in enumerate(periodos):
                tabla.append({
                    'periodo'   : p,
                    'ventas'    : round(serie.iloc[i], 2),
                    'pronostico': fitted_sel[i] if fitted_sel[i] != "" else "—"
                })

            # ---- etiquetas periodos futuros ----
            labels_futuros = [f"P{len(periodos) + i + 1}" for i in range(periodos_futuros)]

            resultados[producto] = {
                'tabla'         : tabla,
                'proyeccion'    : predict_sel,
                'labels_futuros': labels_futuros,
                'historico'     : serie.tolist(),
                'periodos'      : periodos
            }

        return render_template("pronostico.html",
                               resultados=resultados,
                               resumen=resumen,
                               N=N,
                               metodo=metodo,
                               periodos_futuros=periodos_futuros)

    return render_template("pronostico.html")


if __name__ == "__main__":
    app.run(debug=True)