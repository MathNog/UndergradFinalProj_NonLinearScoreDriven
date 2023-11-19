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

ena = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga = CSV.read(path_series*"carga_limpo.csv", DataFrame)
carga_marina = CSV.read(path_series*"dados_cris.csv EMT_rural_cativo.csv", DataFrame)

dict_series = Dict()
dict_series["ena"] = Dict()
dict_series["carga"] = Dict()
dict_series["carga_marina"] = Dict()

dict_series["ena"]["values"] = ena[:,:ENA]
dict_series["ena"]["dates"] = ena[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"] = carga[:,:Data]

valores = parse.(Float64, replace.(carga_marina[:,:value], ","=>"."))

dict_series["carga_marina"]["values"] = valores
dict_series["carga_marina"]["dates"] = carga_marina[:, :timestamp]


" ----- GAS-CNO LogNormal ------ "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "carga"
y = log.(dict_series[serie]["values"])
# y = log.(collect(1:141) .+ rand(Normal(0,10),141))
dates = dict_series[serie]["dates"]

y_norm = FuncoesTeste.normalize_data(y)

steps_ahead = 12
len_train = length(y) - steps_ahead

y_train = y_norm[1:len_train]
y_test = y_norm[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "LogNormal"
dist = UnobservedComponentsGAS.NormalDistribution(missing, missing)
combination = "additive"

d   = 1.0
α   = 0.2
tol = 0.005
stochastic = true

DICT_MODELS["LogNormal"] = Dict() 

DICT_MODELS["LogNormal"]["carga"]=UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, stochastic, combination)

DICT_MODELS["LogNormal"]["ena"]=UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false), 
                                                            Dict(1=>false), Dict(1 => 1), 
                                                            Dict(1 => 12), false, stochastic, combination)

DICT_MODELS["LogNormal"]["carga_marina"]=UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, stochastic, combination)

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]
fitted_model, initial_values_dict = UnobservedComponentsGAS.fit(gas_model, y_train; α=α, tol=tol);

# fitted_model_auto = UnobservedComponentsGAS.auto_gas(gas_model, y_train, 12)

std_residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, true)
residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, false)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious; combination=combination)

fitted_model.fit_in_sample = FuncoesTeste.denormalize_data(fitted_model.fit_in_sample, y)
y_train = FuncoesTeste.denormalize_data(y_train, y)
y_test = FuncoesTeste.denormalize_data(y_test, y)
forecast["mean"] = FuncoesTeste.denormalize_data(forecast["mean"], y)

" ---- Visualizando os resíduos, fit in sample e forecast ----- "

recover_scale = true

recover_scale ? scale="Original" : scale="Log"

path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\$scale\\"

df_hyperparams = DataFrame("d"=>d, "tol"=>tol, "α"=>α)
CSV.write(path_saida*"$(serie)_hyperparams.csv",df_hyperparams)

dict_params = DataFrame(FuncoesTeste.get_parameters(fitted_model))
CSV.write(path_saida*"$(serie)_params.csv",dict_params)

FuncoesTeste.plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

FuncoesTeste.plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

df_forecast_quantiles = FuncoesTeste.get_forecast_quantiles(forecast, [1,5,12])
CSV.write(path_saida*"$(serie)_forecast_quantiles.csv",df_forecast_quantiles)

FuncoesTeste.plot_forecast_histograms(fitted_model, forecast, residuals, distribution, serie, 20, recover_scale)
savefig(path_saida*"$(serie)_forecast_histograms_$(distribution).png")

FuncoesTeste.plot_fit_forecast(fitted_model, forecast, dates_train, y_train, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_fit_forecast_$(distribution).png")

FuncoesTeste.plot_residuals(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

FuncoesTeste.plot_acf_residuals(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

FuncoesTeste.plot_residuals_histogram(std_residuals,distribution, serie)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = FuncoesTeste.get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = FuncoesTeste.get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

FuncoesTeste.plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_components_$(distribution).png")

FuncoesTeste.plot_qqplot(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

FuncoesTeste.plot_diagnosis(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

mapes = FuncoesTeste.get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)


