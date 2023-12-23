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

ena          = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga        = CSV.read(path_series*"carga_limpo.csv", DataFrame)
carga_marina = CSV.read(path_series*"dados_cris.csv EMT_rural_cativo.csv", DataFrame)
airline      = CSV.read(path_series*"AirPassengers.csv", DataFrame)

dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["carga_marina"] = Dict()
dict_series["airline"]      = Dict()

dict_series["ena"]["values"]   = ena[:,:ENA]
dict_series["ena"]["dates"]    = ena[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"]  = carga[:,:Data]

valores = parse.(Float64, replace.(carga_marina[:,:value], ","=>"."))

dict_series["carga_marina"]["values"] = valores
dict_series["carga_marina"]["dates"] = carga_marina[:, :timestamp]

dict_series["airline"]["values"] = airline[:,2] .* 1.0
dict_series["airline"]["dates"]  = airline[:,:Month]

dict_d = Dict(0.0 => "d_0", 0.5 => "d_05", 1.0 => "d_1")

" -------------------- GAS-CNO Gamma -------------------- "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "ena"
y = dict_series[serie]["values"]
dates = dict_series[serie]["dates"]

steps_ahead = 12
len_train = length(y) - steps_ahead

y_ref = y[1:len_train]
y_train = y[1:len_train]
y_test = y[len_train+1:end]

min_val = 1.
max_val = 2.

# y_train = FuncoesTeste.normalize_data(y_train) #airline, carga
y_train = FuncoesTeste.scale_data(y_train, min_val, max_val) #ena
# y_train = FuncoesTeste.scale_data(y_train, 0.1, 1.1) #carga

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "Gamma"
dist = UnobservedComponentsGAS.GammaDistribution(missing, missing)
combination = "multiplicative2"
combinacao = "mult2"

d   = 0.0
α   = 0.9
tol = 0.005
stochastic = true

DICT_MODELS["Gamma"] = Dict() 

DICT_MODELS["Gamma"]["carga"]=UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false),  
                                                        Dict(2=>true),  Dict(2=>false), 
                                                        Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["ena"]=UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false), 
                                                            Dict(2=>false), Dict(2=>1), 
                                                            Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["airline"]=UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false),  
                                                            Dict(2=>true),  Dict(2=>false), 
                                                            Dict(2 => 12), false, stochastic, combination)

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]
fitted_model, initial_values = UnobservedComponentsGAS.fit(gas_model, y_train; α=α, tol=tol, max_optimization_time=240.);

# dict_initial = deepcopy(fitted_model.components["param_2"])
# dict_initial["param_1"] = fitted_model.fitted_params["param_1"][2]
# dict_initial["param_2"] = fitted_model.fitted_params["param_2"][2]

# open("fitted_gamma_ar_sazo_$(combination).json", "w") do f
#     JSON3.write(f, dict_initial)
#     println(f)
# end

# fitted_model.fitted_params["param_1"]

# plot(initial_values["rws"]["values"].+initial_values["slope"]["values"].+initial_values["seasonality"]["values"])
# plot!(y_train)

# gas_model = DICT_MODELS[distribution][serie]
# auto_model = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead)
# fitted_model = auto_model[1]
# gas_model = auto_model[2]
# d = gas_model.d
# α = fitted_model.penalty_factor

std_residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, true)
residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, false)
q_residuals   = FuncoesTeste.get_quantile_residuals(fitted_model)
forecast, dict_hyperparams_and_fitted_components = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious; combination=combination)

# fitted_model.fit_in_sample = FuncoesTeste.denormalize_data(fitted_model.fit_in_sample, y_ref)
# y_train                    = FuncoesTeste.denormalize_data(y_train, y_ref)
# # y_test                     = FuncoesTeste.denormalize_data(y_test, y_ref)
# forecast["mean"]           = FuncoesTeste.denormalize_data(forecast["mean"], y_ref)
# forecast["scenarios"]      = FuncoesTeste.denormalize_data(forecast["scenarios"], y_ref)

fitted_model.fit_in_sample[1] = y_train[1]
fitted_model.fit_in_sample .= FuncoesTeste.unscale_data(fitted_model.fit_in_sample, y_ref)
y_train = FuncoesTeste.unscale_data(y_train, y_ref)
# y_test = FuncoesTeste.unscale_data(y_test, y_ref)
forecast["mean"] = FuncoesTeste.unscale_data(forecast["mean"], y_ref)
forecast["scenarios"] = FuncoesTeste.unscale_data(forecast["scenarios"], y_ref)

# Avaliar possiveis mudancas entre fit e forec
plot(dict_hyperparams_and_fitted_components["ar"]["value"][2,:,:][2:end,1], title = "AR")
plot(dict_hyperparams_and_fitted_components["seasonality"]["value"][2,:,:][2:end,1], title = "Sazo")
" ---- Visualizando os resíduos, fit in sample e forecast ----- "

path_saida = current_path*"\\Saidas\\CombNaoLinear\\$combination\\$(dict_d[d])\\$distribution\\"
# path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\"

recover_scale = false

df_hyperparams = DataFrame("d"=>d, "tol"=>tol, "α"=>α, "stochastic"=>stochastic, "min_val"=>min_val, "max_val"=>max_val)
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

FuncoesTeste.plot_components(fitted_model, dates_train, distribution, "param_2", recover_scale, residuals, serie, combinacao, d)
savefig(path_saida*"$(serie)_components_$(distribution).png")

FuncoesTeste.plot_qqplot(std_residuals, distribution, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

FuncoesTeste.plot_qqplot(q_residuals[2:end], distribution, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_qqplot_$(distribution).png")

FuncoesTeste.plot_diagnosis(std_residuals, dates_train, distribution, true, serie, "pearson", combinacao, d)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

FuncoesTeste.plot_diagnosis(q_residuals[2:end], dates_train, distribution, true, serie, "quantile", combinacao, d)
savefig(path_saida*"$(serie)_quantile_diagnosticos_$(distribution).png")

mapes = FuncoesTeste.get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)
