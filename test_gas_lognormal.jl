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
airline          = CSV.read(path_series*"AirPassengers.csv", DataFrame)

carga_components            = CSV.read(path_series*"components_ets_multiplicativo_carga.csv", DataFrame)[:,2:end]
ena_components              = CSV.read(path_series*"components_ets_multiplicativo_ena.csv", DataFrame)[:,2:end]
carga_components_normalized = CSV.read(path_series*"components_ets_multiplicativo_carga_normalizada.csv", DataFrame)[:,2:end]
ena_components_normalized   = CSV.read(path_series*"components_ets_multiplicativo_ena_normalizada.csv", DataFrame)[:,2:end]

dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["airline"]      = Dict()

dict_series["ena"]["values"]   = ena[:,:ENA]
dict_series["ena"]["dates"]    = ena[:,:Data]
dict_series["ena"]["components"] = ena_components

dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"]  = carga[:,:Data]
dict_series["carga"]["components"] = carga_components

dict_series["airline"]["values"] = airline[:,2] .* 1.0
dict_series["airline"]["dates"]  = airline[:,:Month]

dict_d = Dict(0.0 => "d_0", 0.5 => "d_05", 1.0 => "d_1")

" ----- GAS-CNO LogNormal ------ "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "carga"
y     = log.(dict_series[serie]["values"])
dates = dict_series[serie]["dates"]
initial_components = dict_series[serie]["components"]

steps_ahead = 12
len_train   = length(y) - steps_ahead

y_train = y[1:len_train]
y_test  = y[len_train+1:end]
y_ref   = y[1:len_train]

# min_val = 1.1
# max_val = 2.1

# y_train = FuncoesTeste.normalize_data(y_train) #airline, carga
# y_train = FuncoesTeste.scale_data(y_train, min_val, max_val) #ena
# y_train = FuncoesTeste.scale_data(y_train, 0.1, 1.1) #carga

dates_train = dates[1:len_train]
dates_test  = dates[len_train+1:end]

distribution = "LogNormal"
dist         = UnobservedComponentsGAS.NormalDistribution(missing, missing)
combination  = "multiplicative1"
combinacao   = "mult1"

d   = 0.5
α   = 0.0
tol = 5e-3
stochastic = true

DICT_MODELS["LogNormal"] = Dict() 

DICT_MODELS["LogNormal"]["carga"] = UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, stochastic, combination)

DICT_MODELS["LogNormal"]["ena"] = UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false), 
                                                            Dict(1=>false), Dict(1 => 1), 
                                                            Dict(1 => 12), false, stochastic, combination)

DICT_MODELS["LogNormal"]["airline"] = UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false),  
                                                            Dict(1 => true),  Dict(1 => false), 
                                                            Dict(1 => 12), false, stochastic, combination)

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]

initial_values = FuncoesTeste.get_initial_values_from_components(y_train, initial_components, stochastic, serie, distribution) 

fitted_model, initial_values = UnobservedComponentsGAS.fit(gas_model, y_train; α=α, tol=tol, 
                                                        max_optimization_time=300., initial_values=initial_values);


# dict_initial = deepcopy(fitted_model.components["param_1"])
# dict_initial["param_1"] = fitted_model.fitted_params["param_1"][2]
# dict_initial["param_2"] = fitted_model.fitted_params["param_2"][2]

# open("fitted_lognormal_rws_sazo_$(combination).json", "w") do f
#     JSON3.write(f, dict_initial)
#     println(f)
# end

# gas_model = DICT_MODELS[distribution][serie]
# auto_model = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead; d_values=[d])
# fitted_model = auto_model[1]
# gas_model = auto_model[2]
# d = gas_model.d
# α = fitted_model.penalty_factor

std_residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, true)
residuals     = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, false)
q_residuals   = FuncoesTeste.get_quantile_residuals(fitted_model)
forecast, dict_hyperparams_and_fitted_components = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious; combination=combination)

# fitted_model
# plot(dict_hyperparams_and_fitted_components["rws"]["value"][:,:,1][1,:])
# plot(dict_hyperparams_and_fitted_components["params"][:,:,1][1,:])

# plot(FuncoesTeste.normalize_data(y_test))
# plot!(forecast["mean"])

# fitted_model.fit_in_sample = FuncoesTeste.denormalize_data(fitted_model.fit_in_sample, y_ref)
# y_train                    = FuncoesTeste.denormalize_data(y_train, y_ref)
# forecast["mean"]           = FuncoesTeste.denormalize_data(forecast["mean"], y_ref)
# forecast["scenarios"]      = FuncoesTeste.denormalize_data(forecast["scenarios"], y_ref)

# fitted_model.fit_in_sample[1] = y_train[1]
# fitted_model.fit_in_sample    .= FuncoesTeste.unscale_data(fitted_model.fit_in_sample, y_ref)
# y_train                       = FuncoesTeste.unscale_data(y_train, y_ref)
# forecast["mean"]              = FuncoesTeste.unscale_data(forecast["mean"], y_ref)
# forecast["scenarios"]         = FuncoesTeste.unscale_data(forecast["scenarios"], y_ref)

" ---- Visualizando os resíduos, fit in sample e forecast ----- "

recover_scale = true

recover_scale ? scale="Original" : scale="Log"

path_saida = current_path*"\\Saidas\\CombNaoLinear\\ETS_init\\$combination\\$(dict_d[d])\\$distribution\\"
# path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\Original\\"

df_hyperparams = DataFrame("d"=>d, "tol"=>tol, "α"=>α, "stochastic"=>stochastic)#, "min_val"=>min_val, "max_val"=>max_val)
CSV.write(path_saida*"$(serie)_hyperparams.csv",df_hyperparams)

dict_params = DataFrame(FuncoesTeste.get_parameters(fitted_model))
CSV.write(path_saida*"$(serie)_params.csv",dict_params)

FuncoesTeste.plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, residuals, serie, combinacao, d)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

FuncoesTeste.plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale, serie, combinacao, d)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

df_forecast_quantiles = FuncoesTeste.get_forecast_quantiles(forecast, [1,5,12])
CSV.write(path_saida*"$(serie)_forecast_quantiles.csv",df_forecast_quantiles)

FuncoesTeste.plot_forecast_histograms(fitted_model, forecast, residuals, distribution, serie, 20, recover_scale, combinacao, d)
savefig(path_saida*"$(serie)_forecast_histograms_$(distribution).png")

FuncoesTeste.plot_fit_forecast(fitted_model, forecast, dates_train, y_train, y_test, dates_test, distribution, residuals, recover_scale, serie, combinacao, d)
savefig(path_saida*"$(serie)_fit_forecast_$(distribution).png")

FuncoesTeste.plot_residuals(std_residuals, dates_train, distribution, true, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

FuncoesTeste.plot_residuals(q_residuals[2:end], dates_train[2:end], distribution, false, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_$(distribution).png")

FuncoesTeste.plot_acf_residuals(std_residuals, distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

FuncoesTeste.plot_acf_residuals(q_residuals, distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_acf_$(distribution).png")

FuncoesTeste.plot_residuals_histogram(std_residuals,distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

FuncoesTeste.plot_residuals_histogram(std_residuals,distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = FuncoesTeste.get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = FuncoesTeste.get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

q_residuals_diagnostics_05 = FuncoesTeste.get_residuals_diagnostics(q_residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_quantile_residuals_diagnostics_05.csv",q_residuals_diagnostics_05)

q_residuals_diagnostics_01 = FuncoesTeste.get_residuals_diagnostics(q_residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_quantile_residuals_diagnostics_01.csv",q_residuals_diagnostics_01)

FuncoesTeste.plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals, serie, combinacao, d)
savefig(path_saida*"$(serie)_components_$(distribution).png")

FuncoesTeste.plot_qqplot(std_residuals, distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

FuncoesTeste.plot_qqplot(q_residuals[2:end], distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_qqplot_$(distribution).png")

FuncoesTeste.plot_diagnosis(std_residuals, dates_train, distribution, true, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

FuncoesTeste.plot_diagnosis(q_residuals[2:end], dates_train, distribution, false, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_diagnosticos_$(distribution).png")

mapes = FuncoesTeste.get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)