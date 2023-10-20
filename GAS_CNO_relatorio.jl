using UnobservedComponentsGAS

"Codigo para definição e estimação do modelo GAS Normal"
dist = UnobservedComponentsGAS.NormalDistribution(missing, missing)
time_varying_params = [true, false]
random_walk = Dict(1=>false,2=>false)
random_walk_slope = Dict(1=>true,2=>false)
ar = Dict(1=>false,2=>false)
seasonality = Dict(1=>12)
robust = false
stochastic = false
d = 1.0
num_scenarious = 500

gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust,stochastic)

fitted_model = UnobservedComponentsGAS.fit(gas_model, y_fit)
fitted_model = UnobservedComponentsGAS.auto_gas(gas_model, y_fit, steps_ahead)

residuals = fitted_model.residuals
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_fit, steps_ahead, num_scenarious)