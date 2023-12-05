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

" -------------------- GAS-CNO Gamma -------------------- "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "carga"
y = dict_series[serie]["values"]
dates = dict_series[serie]["dates"]

y_norm = FuncoesTeste.normalize_data(y)

steps_ahead = 12
len_train = length(y) - steps_ahead

y_train = y_norm[1:len_train]
y_test = y_norm[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "Gamma"
dist = UnobservedComponentsGAS.GammaDistribution(missing, missing)
combination = "additive"

d   = 1.0
α   = 0.5
tol = 0.005
stochastic = false

DICT_MODELS["Gamma"] = Dict() 

DICT_MODELS["Gamma"]["carga"]=UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false),  
                                                        Dict(2=>true),  Dict(2=>false), 
                                                        Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["ena"]=UnobservedComponentsGAS.GASModel(dist, [false, true], d, Dict(2=>false), 
                                                            Dict(2=>false), Dict(2=>2), 
                                                            Dict(2 => 12), false, stochastic, combination)

DICT_MODELS["Gamma"]["carga_marina"]=UnobservedComponentsGAS.GASModel(dist, [true, false], d, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, stochastic, combination)

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]
fitted_model, initial_values = UnobservedComponentsGAS.fit(gas_model, y_train; α=α, tol=tol, max_optimization_time=240.);

plot(initial_values["rws"]["values"].+initial_values["slope"]["values"].+initial_values["seasonality"]["values"].+initial_values["intercept"]["values"])

# gas_model = DICT_MODELS[distribution][serie]
# auto_model = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead)
# fitted_model = auto_model[1]
# gas_model = auto_model[2]
# d = gas_model.d
# α = fitted_model.penalty_factor

std_residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, true)
residuals = FuncoesTeste.get_residuals(fitted_model, distribution, y_train, false)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious; combination=combination)

fitted_model.fit_in_sample = FuncoesTeste.denormalize_data(fitted_model.fit_in_sample, y)
y_train = FuncoesTeste.denormalize_data(y_train, y)
y_test = FuncoesTeste.denormalize_data(y_test, y)
forecast["mean"] = FuncoesTeste.denormalize_data(forecast["mean"], y)
forecast["scenarios"] = FuncoesTeste.denormalize_data(forecast["scenarios"], y)

" ---- Visualizando os resíduos, fit in sample e forecast ----- "

path_saida = current_path*"\\Saidas\\Benchmark\\2parametros\\$distribution\\"
recover_scale = false

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



" AutoARIMA Benchmark"

# path_saida = current_path*"\\Saidas\\Benchmark\\2parametros\\AutoARIMA\\"

# dict_benchmarks          = Dict()
# dict_benchmarks["carga"] = Dict()
# dict_benchmarks["ena"]   = Dict()
# dict_benchmarks["vazao"] = Dict()

# dict_benchmarks["carga"]["fit"]      = CSV.read(path_series*"carga_fit_autoarima.csv",DataFrame)
# dict_benchmarks["carga"]["forecast"] = CSV.read(path_series*"carga_forecast_autoarima.csv",DataFrame)

# dict_benchmarks["ena"]["fit"]      = CSV.read(path_series*"ena_fit_autoarima.csv",DataFrame)
# dict_benchmarks["ena"]["forecast"] = CSV.read(path_series*"ena_forecast_autoarima.csv",DataFrame)

# dict_benchmarks["vazao"]["fit"]      = CSV.read(path_series*"vazao_fit_autoarima.csv",DataFrame)
# dict_benchmarks["vazao"]["forecast"] = CSV.read(path_series*"vazao_forecast_autoarima.csv",DataFrame)


# for serie in ["carga", "ena", "vazao"]
    
#     y = dict_series[serie]["values"]
#     dates = dict_series[serie]["dates"]
#     fit = dict_benchmarks[serie]["fit"].Values
#     dates_fit = dict_benchmarks[serie]["fit"].Data
#     prev = dict_benchmarks[serie]["forecast"].Values
#     dates_prev = dict_benchmarks[serie]["forecast"].Data

#     residuals = y[1:end-12] .- fit
#     std_residuals = (residuals .- mean(residuals))./std(residuals)

#     mape_train = round(100*MAPE(y[1:end-12],fit), digits=2)
#     mape_test = round(100*MAPE(y[end-11:end],prev), digits=2)
#     df_mapes = DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
#     CSV.write(path_saida*"$(serie)_mapes.csv",df_mapes)

#     plot(title="Fit in sample AutoARIMA $serie")
#     plot!(dates, y, label="Série")
#     plot!(dates_fit, fit, label="Fit: MAPE = $mape_train%")
#     savefig(path_saida*"$(serie)_fit_autoarima.png")

#     plot(title="Forecast AutoARIMA $serie")
#     plot!(dates[end-11:end], y[end-11:end], label="Série")
#     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
#     savefig(path_saida*"$(serie)_forecast_autoarima.png")

#     plot(title="Fit and Forecast AutoARIMA $serie")
#     plot!(dates, y, label="Série")
#     plot!(dates_fit, fit, label="Fit; MAPE = $mape_train%")
#     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
#     savefig(path_saida*"$(serie)_fit_forecast_autoarima.png")

#     qq = plot(qqplot(Normal, std_residuals))
#     qq = plot!(title="QQPlot Residuos")
#     h = histogram(std_residuals, title="Histograma Residuos", label="")
#     r = plot(title="Resíduos Padronizados")
#     r = plot!(dates_fit, std_residuals , label="Resíduos")
#     acf_values = autocor(std_residuals)
#     lag_values = collect(0:length(acf_values) - 1)
#     conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval
#     a = plot(title="FAC dos Residuos")
#     a = plot!(autocor(std_residuals),seriestype=:stem, label="")
#     a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
#     plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
#         plot_title = "Diagnosticos Residuos AutoARIMA - $serie", title=["Resíduos Padronizados" "FAC dos Residuos" "Histograma Residuos" "QQPlot Residuos"])
#     savefig(path_saida*"$(serie)_diagnosticos_autoarima.png")    
# end

