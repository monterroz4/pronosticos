import pandas as pd

datos=pd.read_csv ("venta_historicas.csv")

# funcion para ejecutar pronostico

def pronosticar(historia,N):
  datos["Pronostico"]=datos["ventas"].rolling(window=N).mean().shift(1)
  datos["error"]=datos["Pronostico"] - datos["ventas"]
  datos["error_abs"]=datos["error"].abs()
  datos["ape"]=datos["error_abs"]/datos["ventas"]
  datos["ape'"]=datos["error_abs"]/datos["Pronostico"]
  datos["error_cuadrado"]=datos["error"]**2

  #medidas de error
  MAPE=datos["ape"].mean()
  MAPE_prima=datos["ape'"].mean()
  MSE=datos["error_cuadrado"].mean()
  RMSE=MSE**0,5


  return datos,MAPE,MAPE_prima,MSE,RMSE

print(pronosticar(datos,3))



