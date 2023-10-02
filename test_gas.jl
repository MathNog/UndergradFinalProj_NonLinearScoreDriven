using JuMP, Ipopt, CSV, DataFrames, Dates, Parameters, Plots, MLJBase, Statistics
using StatsBase, UnPack
using ARCHModels, HypothesisTests
using StatsPlots, Distributions, SpecialFunctions

import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")


function MAPE(A, F)
    return MLJBase.mape(F,A)
end

function Γ(x)
    return SpecialFunctions.gamma(x)
end

current_path = pwd()

" ------------- Funções Auxiliares --------------"


function correct_scale(series, K, residuals)
    SQR = sum(residuals.^2)
    N = length(residuals)
    σ² = SQR/(N-K)
    return exp.(series)*exp(0.5*σ²)
end

function get_number_parameters(fitted_model)
    K = 0
    for param in keys(fitted_model.components)
        for (key,value) in fitted_model.components[param]
            if key != "intercept"
                for (key2, value2) in value["hyperparameters"]
                    K+=length(value2)
                end
            else
                K+=1
            end
        end
    end
    return K
end

function get_residuals(fitted_model, model, y)
    if model == "Normal"
        return fitted_model.residuals["std_residuals"]
    else
        return y .- fitted_model.fit_in_sample
    end
end

function plot_residuals(residuals, dates, model, std_bool, serie)
    std_bool==true ? res = (residuals.-mean(residuals))./std(residuals) : res = residuals
    std_bool==true ? std_title = "Padronizados" : std_title = ""
    plot(title="Resíduos $std_title $model - $serie")
    plot!(dates[2:end], res[2:end] , label="Resíduos")
end

function plot_acf_residuals(residuals, model, serie)
    
    acf_values = autocor(residuals[2:end])
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval

    plot(title="FAC dos Residuos $model - $serie")
    plot!(autocor(residuals[2:end]),seriestype=:stem, label="")
    hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
end

function get_residuals_diagnosis_pvalues(residuals, fitted_model)
    dof = get_number_parameters(fitted_model)
    # Jarque Bera
    jb = pvalue(JarqueBeraTest(residuals))
    # Ljung Box
    lb = pvalue(LjungBoxTest(residuals, 24, dof))
    # ARCH
    arch = pvalue(ARCHLMTest(residuals, 12))
    
    return Dict(:lb=>lb, :jb=>jb, :arch=>arch)
end

function get_residuals_diagnostics(residuals, α, fitted_model)
    d = get_residuals_diagnosis_pvalues(residuals[2:end], fitted_model)
    nomes = Dict(:lb=>"Ljung Box", :jb=>"Jarque Bera", :arch=>"ARCHLM")
    rows = []
    for (teste,pvalue) in d
        nome = nomes[teste]
        pvalue<α ? rejeicao = "Rejeita H₀" : rejeicao = "Não rejeita H₀"
        push!(rows, (nome, α, pvalue, rejeicao))
    end
    df= DataFrame(rows)
    rename!(df, ["Teste", "$α", "pvalor", "Rejeicao"])
    return df
end

function plot_residuals_histogram(residuals, model, serie)
    histogram(residuals[2:end], title="Histograma Residuos $model - $serie", label="")
end

function plot_fit_in_sample(fitted_model, fit_dates, y_train, model, recover_scale, serie)
    
    fit_in_sample = fitted_model.fit_in_sample[2:end]
    if recover_scale
        y_train = exp.(y_train)
        K = get_number_parameters(fitted_model)
        fit_in_sample = correct_scale(fit_in_sample, K, residuals)
    end

    plot(fit_dates[2:end], y_train[2:end], label="Série")
    plot!(fit_dates[2:end], fit_in_sample[1:end], label="Fit in sample")    
    plot!(title=" Fit in sample GAS-CNO $model - $serie")
    
end

function plot_forecast(fitted_model, forecast, y_test, forecast_dates, model, residuals, recover_scale, serie)

    if recover_scale
        y_test = exp.(y_test)
        K = get_number_parameters(fitted_model)
        forecast_mean = correct_scale(forecast["mean"], K, residuals)    
    else
        forecast_mean = forecast["mean"]
    end
    p = plot(title = "Forecast GAS-CNO $model - $serie")
    p = plot!(forecast_dates, y_test, label="Série")
    p = plot!(forecast_dates, forecast_mean, label="Forecast", color="red")
    display(p)
end

function get_components(fitted_model, param, recover_scale, residuals)    
    dict_components = fitted_model.components[param]
    components = Dict()
    if recover_scale
        K = get_number_parameters(fitted_model)
        for key in keys(dict_components)
            key != "intercept" ? components[key] = correct_scale(dict_components[key]["value"], K, residuals) : nothing
        end
            
    else     
        for key in keys(dict_components)
            key != "intercept" ? components[key] = dict_components[key]["value"] : nothing
        end
    end
    return components
end

function plot_components(fitted_model, estimation_dates, model, param, recover_scale, residuals, serie)
    components = get_components(fitted_model, param, recover_scale, residuals)
    
    "level" in keys(components) ? level = components["level"] : level = ones(length(estimation_dates)).*missing
    "slope" in keys(components) ? slope = components["slope"] : slope = ones(length(estimation_dates)).*missing
    "seasonality" in keys(components) ? seasonality = components["seasonality"] : seasonality = ones(length(estimation_dates)).*missing

    p1 = plot(estimation_dates[2:end], level[2:end], label="Level")
    p2 = plot(estimation_dates[2:end],slope[2:end], label="Slope")
    p3 = plot(estimation_dates[2:end], seasonality[2:end], label="Seasonality")
    plot(p1, p2, p3, layout = (3,1) ,plot_title = "Componentes GAS-CNO $model - $serie")#tirei o $param por hora, dado que estou usando apenas 1 parametro variante
end

function plot_qqplot(residuals, model, serie)
    plot(qqplot(Normal, residuals))
    plot!(title="QQPlot Residuos $model - $serie")
end

function get_mapes(y_train, y_test, fitted_model, forecast, residuals, recover_scale)
    
    fit_in_sample = fitted_model.fit_in_sample[2:end]
    forecast_mean = forecast["mean"]
    if recover_scale
        y_train = exp.(y_train)
        y_test = exp.(y_test)
        K = get_number_parameters(fitted_model)
        fit_in_sample = correct_scale(fit_in_sample, K, residuals)
        forecast_mean = correct_scale(forecast["mean"], K, residuals)
    end
    mape_train = 100*MLJBase.mape(fit_in_sample, y_train[2:end])
    mape_test = 100*MLJBase.mape(forecast_mean, y_test)
    
    return DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
end


" ------- Criando dicionario de Dados ------- "

path_series = current_path*"\\Dados\\Tratados\\"

ena = CSV.read(path_series*"ena_limpo.csv",DataFrame)
vazao = CSV.read(path_series*"vazao_limpo.csv", DataFrame)
carga = CSV.read(path_series*"carga_limpo.csv", DataFrame)

dict_series = Dict()
dict_series["ena"] = Dict()
dict_series["vazao"] = Dict()
dict_series["carga"] = Dict()

dict_series["ena"]["values"] = ena[:,:ENA]
dict_series["ena"]["dates"] = ena[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"] = carga[:,:Data]
dict_series["vazao"]["values"] = round.(vazao[:,:Vazao],digits=2)
dict_series["vazao"]["dates"] = vazao[:,:Data]


p1 = plot(dict_series["vazao"]["dates"],dict_series["vazao"]["values"],label="")
p2 = histogram(dict_series["vazao"]["values"], bins=25,label="")
plot(p1,p2, layout = (2,1) , plot_title="Série de Vazão e seu histograma")
savefig(current_path*"\\Saidas\\Relatorio\\SeriesTestes\\vazao.png")

" ----- GAS-CNO LogNormal ------ "

serie = "vazao"
y = log.(dict_series[serie]["values"])
dates = dict_series[serie]["dates"]

steps_ahead = 12
len_train = length(y) - steps_ahead

y_train = y[1:len_train]
y_test = y[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "LogNormal"
dist = UnobservedComponentsGAS.NormalDistribution(missing, missing)
time_varying_params = [true, false]
random_walk = Dict(2 => false, 1=>false)
random_walk_slope = Dict(1 => true, 2=>false)
ar = Dict(1 => false, 2 => false)
seasonality = Dict(1 => 12)
robust = false
stochastic = false
d = 1.0
α = 0.5
num_scenarious = 500

gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)
fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train)

# gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)
# auto_gas_output = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead)
# fitted_model = auto_gas_output[1]

residuals = get_residuals(fitted_model, distribution, y_train)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious)
        
" ---- Visualizando os resíduos, fit in sample e forecast ----- "

recover_scale = true

recover_scale ? scale="Original" : scale="Log"

path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\$scale\\"

plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, serie)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

plot_residuals(residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

plot_acf_residuals(residuals, distribution, serie)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

plot_residuals_histogram(residuals,distribution, serie)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_components_$(distribution).png")

plot_qqplot(residuals, distribution, serie)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

mapes = get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)

" -------------------- GAS-CNO Gamma -------------------- "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

y = dict_series["carga"]["values"]
dates = dict_series["carga"]["dates"]

steps_ahead = 1
len_train = length(y) - steps_ahead

y_train = y[1:len_train]
y_test = y[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "Gamma"
dist = UnobservedComponentsGAS.GammaDistribution(missing, missing)
time_varying_params = [true, false] # apenas o λ varia no tempo
random_walk = Dict(1=>false)
random_walk_slope = Dict(1=>true)
ar = Dict(1=>false)
seasonality = Dict(1=>12)
robust = false
d = 1.0
α = 0.5
stochastic = false
num_scenarious = 500

gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)
fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train)

gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)
auto_gas_output = UnobservedComponentsGAS.auto_gas(gas_model, y_train, 12)
auto_gas_output
fitted_model = auto_gas_output[1]

residuals = get_residuals(fitted_model, distribution, y_train)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious)


" ---- Visualizando os resíduos, fit in sample e forecast ----- "
path_saida = current_path*"\\Saidas\\Benchmark\\$distribution\\"
recover_scale = false
plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution)_carga.png")

plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale)
savefig(path_saida*"$(serie)_forecast_$(distribution)_carga.png")

plot_residuals(residuals, dates_train, distribution, true)
savefig(path_saida*"$(serie)_residuals_$(distribution)_carga.png")

plot_acf_residuals(residuals, distribution)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution)_carga.png")

plot_residuals_histogram(residuals,distribution)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution)_carga.png")

residuals_diagnostics_05 = get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals)
savefig(path_saida*"$(serie)_components_$(distribution)_carga.png")

plot_qqplot(residuals, distribution)
savefig(path_saida*"$(serie)_qqplot_$(distribution)_carga.png")


