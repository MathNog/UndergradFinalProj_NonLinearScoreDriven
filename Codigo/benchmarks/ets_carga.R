library(forecast)

normalize_data = function(y){
  min_y = min(y)
  max_y = max(y)
  return(((y-min_y) / (max_y-min_y))+1)
}

" -------- S�rie de Carga ------------"

carga = read.csv("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\carga_limpo.csv")

y = carga$Carga
datas = carga$Data

len_test = 12
len_train = length(y) - len_test
datas_train = datas[1:len_train]

" ---- y em n�vel ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(2002,1), end = c(2022, 6), frequency = 12)

modelo = ets(y_train, model = "MAM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_carga.csv")


" ---- y em log ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(2002,1), end = c(2022, 6), frequency = 12)

modelo = ets(y_train, model = "MAM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_carga_log.csv")



" --- y normalizado ---"

y_train = normalize_data(y[1:len_train])
y_train = ts(y_train, start = c(2002,1), end = c(2022, 6), frequency = 12)

modelo = ets(y_train, model = "MAM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_carga_normalizada.csv")


" ---- y em n�vel aditivo ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(2002,1), end = c(2022, 6), frequency = 12)

modelo = ets(y_train, model = "AAA")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_aditivo_carga.csv")


" ---- y em log aditivo ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(2002,1), end = c(2022, 6), frequency = 12)

modelo = ets(y_train, model = "AAA")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_aditivo_carga_log.csv")



" -------- S�rie de ENA ------------"

ena = read.csv("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\ena_limpo.csv")

y = ena$ENA
datas = ena$Data

len_test = 12
len_train = length(y) - len_test
datas_train = datas[1:len_train]

" ---- y em n�vel ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(2000,1), end = c(2022, 8), frequency = 12)

modelo = ets(y_train, model = "MNM")
#level = modelo$states[1:len_train, "l"]
#sazo = modelo$states[1:len_train, c(2:13)]

write.csv(modelo$states, file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_ena.csv")

" ---- y em log ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(2000,1), end = c(2022, 8), frequency = 12)

modelo = ets(y_train, model = "MNM")
#level = modelo$states[1:len_train, "l"]
#sazo = modelo$states[1:len_train, c(2:13)]

write.csv(modelo$states, file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_ena_log.csv")


" --- y normalizado ---"

y_train = normalize_data(y[10:len_train])
y_train = ts(y_train, start = c(2000,1), end = c(2022, 8), frequency = 12)

modelo = ets(y_train, model = "MNM")
level = modelo$states[1:len_train, "l"]
sazo = modelo$states[1:len_train, c(2:13)]

write.csv(modelo$states, file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_ena_normalizada.csv")



" ---- y em n�vel aditivo ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(2000,1), end = c(2022, 8), frequency = 12)

modelo = ets(y_train, model = "ANA")
#level = modelo$states[1:len_train, "l"]
#sazo = modelo$states[1:len_train, c(2:13)]

write.csv(modelo$states, file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_aditivo_ena.csv")

" ---- y em log aditivo ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(2000,1), end = c(2022, 8), frequency = 12)

modelo = ets(y_train, model = "ANA")
#level = modelo$states[1:len_train, "l"]
#sazo = modelo$states[1:len_train, c(2:13)]

write.csv(modelo$states, file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_aditivo_ena_log.csv")

" -------- S�rie de UK Visits ------------"

uk_visits = read.csv("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\uk_visits.csv")

y = uk_visits$Valor
datas = uk_visits$Data

len_test = 12
len_train = length(y) - len_test
datas_train = datas[1:len_train]

" ---- y em n�vel ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(1980,1), end = c(2005, 12), frequency = 12)

modelo = ets(y_train, model = "MAM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_uk_visits.csv")


" ---- y em log ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(1980,1), end = c(2005, 12), frequency = 12)

modelo = ets(y_train, model = "MAM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_uk_visits_log.csv")




" -------- S�rie de Precipita��o ------------"

precipitacao = read.csv("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\uhe_belo_monte_limpo.csv")

y = precipitacao$MEDIA
datas = precipitacao$DATA

len_test = 12
len_train = length(y) - len_test
datas_train = datas[1:len_train]

" ---- y em n�vel ---"

y_train = y[1:len_train]
y_train = ts(y_train, start = c(2000,6), end = c(2021, 10), frequency = 12)

modelo = ets(y_train, model = "MNM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

plot(modelo)

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_precipitacao.csv")


" ---- y em log ---"

y_train = log(y[1:len_train])
y_train = ts(y_train, start = c(2000,6), end = c(2021, 10), frequency = 12)

modelo = ets(y_train, model = "MNM")
level = modelo$states[1:len_train, "l"]
slope = modelo$states[1:len_train, "b"]
sazo = modelo$states[1:len_train, c(3:14)]

write.csv(modelo$states[1:len_train,], file="C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Per�odos\\TCC\\Dados\\Tratados\\components_ets_multiplicativo_precipitacao_log.csv")


