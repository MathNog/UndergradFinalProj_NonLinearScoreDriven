using JuMP, Ipopt, CSV, DataFrames, Dates, Parameters, Plots, MLJBase, Statistics
using StatsBase, UnPack
using ARCHModels, HypothesisTests
using StatsPlots, Distributions, SpecialFunctions

include("funcoes_teste.jl")
using ..FuncoesTeste

import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

current_path = pwd()

" ------- Criando dicionario de Dados ------- "

path_series = current_path*"\\Dados\\Tratados\\"
DICT_MODELS = Dict()

ena              = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga            = CSV.read(path_series*"carga_limpo.csv", DataFrame)
uk_visits        = CSV.read(path_series*"uk_visits.csv", DataFrame)

carga_components            = CSV.read(path_series*"components_ets_multiplicativo_carga.csv", DataFrame)[:,2:end]
ena_components              = CSV.read(path_series*"components_ets_multiplicativo_ena.csv", DataFrame)[2:end,2:end]
uk_visits_components        = CSV.read(path_series*"components_ets_multiplicativo_uk_visits.csv", DataFrame)[:,2:end]

# ena_components              = CSV.read(path_series*"components_ets_aditivo_ena.csv", DataFrame)[2:end,2:end]

dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["uk_visits"]      = Dict()

dict_series["ena"]["values"]     = ena[:,:ENA]
dict_series["ena"]["dates"]      = ena[:,:Data]
dict_series["ena"]["components"] = ena_components

dict_series["carga"]["values"]     = carga[:,:Carga]
dict_series["carga"]["dates"]      = carga[:,:Data]
dict_series["carga"]["components"] = carga_components

dict_series["uk_visits"]["values"]     = Float64.(uk_visits[:,:Valor])
dict_series["uk_visits"]["dates"]      = uk_visits[:,:Data]
dict_series["uk_visits"]["components"] = uk_visits_components


dict_d = Dict(0.0 => "d_0", 0.5 => "d_05", 1.0 => "d_1")

" -------------------- GAS-CNO Gamma -------------------- "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "uk_visits"
y                  = dict_series[serie]["values"]
dates              = dict_series[serie]["dates"]
initial_components = dict_series[serie]["components"]

steps_ahead = 12
len_train   = length(y) - steps_ahead

y_ref   = y[1:len_train]
y_train = y[1:len_train]
y_test  = y[len_train+1:end]

dates_train = dates[1:len_train]
dates_test  = dates[len_train+1:end]

y_train = FuncoesTeste.normalize_data(y_train)

distribution = "Gamma"
dist         = UnobservedComponentsGAS.GammaDistribution(missing, missing)
combination  = "multiplicative2"
combinacao   = "mult2"

d   = 1.0
α   = 0.0
tol = 5e-3
stochastic = false

DICT_MODELS["Gamma"] = Dict() 

DICT_MODELS["Gamma"]["carga"] = UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false),  
                                                        Dict(2=>true),  Dict(2=>false), 
                                                        Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["ena"] = UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false), 
                                                            Dict(2=>false), Dict(2=>1), 
                                                            Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["uk_visits"] = UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false),  
                                                            Dict(2=>true),  Dict(2=>false), 
                                                            Dict(2 => 12), false, stochastic, combination)

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]

initial_values = FuncoesTeste.get_initial_values_from_components(y_train, initial_components, stochastic, serie, distribution) 

fitted_model, initial_values = UnobservedComponentsGAS.fit(gas_model, y_train; α=α, tol=tol, 
                                                        max_optimization_time=300., initial_values=initial_values);
                                                        
fitted_model.fit_in_sample
fitted_model.fit_in_sample[1] = y_train[1]
var           = fitted_model.fitted_params["param_2"].^2 ./ fitted_model.fitted_params["param_1"]
std_residuals = FuncoesTeste.get_std_residuals(y_train, fitted_model.fit_in_sample, var)#FuncoesTeste.get_residuals(fitted_model, distribution, y_train, true)
q_residuals   = FuncoesTeste.get_quantile_residuals(fitted_model)
res           = y_train .- fitted_model.fit_in_sample
forecast, dict_hyperparams_and_fitted_components = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious; combination=combination);

fitted_model.fit_in_sample = FuncoesTeste.denormalize_data(fitted_model.fit_in_sample, y_ref)
y_train                    = FuncoesTeste.denormalize_data(y_train, y_ref)
forecast["mean"]           = FuncoesTeste.denormalize_data(forecast["mean"], y_ref)
forecast["scenarios"]      = FuncoesTeste.denormalize_data(forecast["scenarios"], y_ref)

" ---- Visualizando os resíduos, fit in sample e forecast ----- "

path_saida = current_path*"\\Saidas\\CombNaoLinear\\SazoDeterministica\\$combination\\$distribution\\"
# path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\"

recover_scale = false

df_hyperparams = DataFrame("d"=>d, "tol"=>tol, "α"=>α, "stochastic"=>stochastic)#, "min_val"=>min_val, "max_val"=>max_val)
CSV.write(path_saida*"$(serie)_hyperparams.csv",df_hyperparams)

dict_params = DataFrame(FuncoesTeste.get_parameters(fitted_model))
CSV.write(path_saida*"$(serie)_params.csv",dict_params)

df_fitted_values = FuncoesTeste.get_fitted_values(fitted_model, dates_train, std_residuals, q_residuals, res, "param_2", recover_scale)
CSV.write(path_saida*"$(serie)_fitted_values.csv",df_fitted_values)

df_forecast = FuncoesTeste.get_forecast_values(forecast, dates_test)
CSV.write(path_saida*"$(serie)_forecast_values.csv",df_forecast)

FuncoesTeste.plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, res, serie, combinacao, d)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

FuncoesTeste.plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, res, recover_scale, serie, combinacao, d)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

df_forecast_quantiles = FuncoesTeste.get_forecast_quantiles(forecast, [1,5,12])
CSV.write(path_saida*"$(serie)_forecast_quantiles.csv",df_forecast_quantiles)

FuncoesTeste.plot_forecast_histograms(fitted_model, forecast, res, distribution, serie, 20, recover_scale, combinacao, d)
savefig(path_saida*"$(serie)_forecast_histograms_$(distribution).png")

FuncoesTeste.plot_fit_forecast(fitted_model, forecast, dates_train, y_train, y_test, dates_test, distribution, res, recover_scale, serie, combinacao, d)
savefig(path_saida*"$(serie)_fit_forecast_$(distribution).png")

log_like = FuncoesTeste.get_log_likelihood(fitted_model)
CSV.write(path_saida*"$(serie)_log_likelihood.csv",DataFrame(Dict("Log Likelihood"=>log_like)))

FuncoesTeste.plot_residuals(std_residuals[2:end], dates_train[2:end], distribution, true, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

FuncoesTeste.plot_residuals(q_residuals[2:end], dates_train[2:end], distribution, false, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_$(distribution).png")

FuncoesTeste.plot_acf_residuals(std_residuals[2:end], distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

FuncoesTeste.plot_acf_residuals(q_residuals[2:end], distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_acf_$(distribution).png")

FuncoesTeste.plot_residuals_histogram(std_residuals[2:end], distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

FuncoesTeste.plot_residuals_histogram(q_residuals[2:end],distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = FuncoesTeste.get_residuals_diagnostics(std_residuals[2:end], 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = FuncoesTeste.get_residuals_diagnostics(std_residuals[2:end], 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

q_residuals_diagnostics_05 = FuncoesTeste.get_residuals_diagnostics(q_residuals[2:end], 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_quantile_residuals_diagnostics_05.csv",q_residuals_diagnostics_05)

q_residuals_diagnostics_01 = FuncoesTeste.get_residuals_diagnostics(q_residuals[2:end], 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_quantile_residuals_diagnostics_01.csv",q_residuals_diagnostics_01)

FuncoesTeste.plot_components(fitted_model, dates_train, distribution, "param_2", recover_scale, res, serie, combinacao, d)
savefig(path_saida*"$(serie)_components_$(distribution).png")

FuncoesTeste.plot_qqplot(std_residuals[2:end], distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

FuncoesTeste.plot_qqplot(q_residuals[2:end], distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_qqplot_$(distribution).png")

FuncoesTeste.plot_diagnosis(std_residuals[2:end], dates_train[2:end], distribution, true, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

FuncoesTeste.plot_diagnosis(q_residuals[2:end], dates_train[2:end], distribution, false, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_diagnosticos_$(distribution).png")

mapes = FuncoesTeste.get_mapes(y_train, y_test, fitted_model, forecast, res ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)