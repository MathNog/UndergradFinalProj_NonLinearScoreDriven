using ScoreDrivenModels

"Codigo para definição e estimação do modelo GAS(p,q) Gama"
gas = ScoreDrivenModel([1, 12], [1, 12], Gamma, 0.0) #definição do modelo
ScoreDrivenModels.fit!(gas,y_fit) #estimação do modelo
fit_in_sample = fitted_mean(gas, y_fit) #obtenção do fit in sample
residuals = y_fit .- fit_in_sample # obtenção dos resíduos
forecast = ScoreDrivenModels.forecast(y_fit, gas, 12) # obtenção das previsões