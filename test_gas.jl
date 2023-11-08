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
                    if key2 in ["γ", "γ_star"]
                        K+=size(value2,1)
                    else
                        K+=length(value2)
                    end
                end
            else
                K+=1
            end
        end
    end
    return K
end

function get_parameters(fitted_model)
    dict_params = Dict()
    for param in keys(fitted_model.components)
        for (key,value) in fitted_model.components[param]
            if key != "intercept"
                for (key2, value2) in value["hyperparameters"]
                    if (key2 !="γ") && (key2!="γ_star")
                        dict_params[key*"_"*key2] = value2
                    end
                end
            end
        end
    end
    return dict_params
end

function get_residuals(fitted_model, model, y, standarize)
    if model == "Normal"
        return fitted_model.residuals["std_residuals"][2:end]
    else
        r = (y .- fitted_model.fit_in_sample)[2:end]
        if standarize
            return (r.-mean(r))./std(r)
        else
            return r
        end
    end
end

function plot_residuals(residuals, dates, model, std_bool, serie)
    # std_bool==true ? res = (residuals.-mean(residuals))./std(residuals) : res = residuals
    std_bool==true ? std_title = "Padronizados" : std_title = ""
    plot(title="Resíduos $std_title $model - $serie")
    plot!(dates[2:end], residuals , label="Resíduos")
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

function plot_fit_in_sample(fitted_model, fit_dates, y_train, model, recover_scale, residuals, serie)
    
    fit_in_sample = fitted_model.fit_in_sample[2:end]
    # println(fit_in_sample)
    if recover_scale
        y_train = exp.(y_train)
        K = get_number_parameters(fitted_model)
        fit_in_sample = correct_scale(fit_in_sample, K, residuals)
    end
    # println(fit_in_sample)
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

function (fitted_model, forecast,fit_dates, y_train, y_test, forecast_dates, model, residuals, recover_scale, serie)
    fit_in_sample = fitted_model.fit_in_sample[2:end]
    dates = vcat(fit_dates, forecast_dates)
    if recover_scale
        y_test = exp.(y_test)
        y_train = exp.(y_train)
        K = get_number_parameters(fitted_model)
        forecast_mean = correct_scale(forecast["mean"], K, residuals)    
        fit_in_sample = correct_scale(fit_in_sample, K, residuals)
    else
        forecast_mean = forecast["mean"]
    end
    y = vcat(y_train, y_test)
    p = plot(title = "Fit and Forecast GAS-CNO $model - $serie")
    p = plot!(dates[2:end], y[2:end], label="Série")
    p = plot!(fit_dates[2:end], fit_in_sample[1:end], label="Fit in sample") 
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

function plot_diagnosis(residuals, dates, model, std_bool, serie)
    qq = plot(qqplot(Normal, residuals))
    qq = plot!(title="QQPlot Residuos")

    h = histogram(residuals, title="Histograma Residuos", label="")

    # std_bool==true ? res = (residuals.-mean(residuals))./std(residuals) : res = residuals
    std_bool==true ? std_title = "Padronizados" : std_title = ""
    r = plot(title="Resíduos $std_title")
    r = plot!(dates[2:end], residuals , label="Resíduos")

    acf_values = autocor(residuals[2:end])
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval

    a = plot(title="FAC dos Residuos")
    a = plot!(autocor(residuals[2:end]),seriestype=:stem, label="")
    a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")

    plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
        plot_title = "Diagnosticos Residuos GAS-CNO $model - $serie", title=["Resíduos $std_title" "FAC dos Residuos" "Histograma Residuos" "QQPlot Residuos"])
    
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
    mape_train = round(100*MLJBase.mape(fit_in_sample, y_train[2:end]), digits=2)
    mape_test = round(100*MLJBase.mape(forecast_mean, y_test), digits=2)
    
    return DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
end

function normalize_data(y::Vector{Fl})where{Fl}
    min = minimum(y)
    max = maximum(y)
    return ((y .- min) ./ (max - min)) .+ 1.
end

function denormalize_data(y_norm::Vector{Fl}, y::Vector{Fl}) where{Fl}
    min = minimum(y)
    max = maximum(y)
    return (y_norm .- 1.) .* (max-min) .+ min
end


" ------- Criando dicionario de Dados ------- "

path_series = current_path*"\\Dados\\Tratados\\"
DICT_MODELS = Dict()

ena = CSV.read(path_series*"ena_limpo.csv",DataFrame)
vazao = CSV.read(path_series*"vazao_limpo.csv", DataFrame)
carga = CSV.read(path_series*"carga_limpo.csv", DataFrame)
carga_marina = CSV.read(path_series*"dados_cris.csv EMT_rural_cativo.csv", DataFrame)

dict_series = Dict()
dict_series["ena"] = Dict()
dict_series["vazao"] = Dict()
dict_series["carga"] = Dict()
dict_series["carga_marina"] = Dict()

dict_series["ena"]["values"] = ena[:,:ENA]
dict_series["ena"]["dates"] = ena[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"] = carga[:,:Data]
dict_series["vazao"]["values"] = round.(vazao[:,:Vazao],digits=2)
dict_series["vazao"]["dates"] = vazao[:,:Data]

valores = parse.(Float64, replace.(carga_marina[:,:value], ","=>"."))

dict_series["carga_marina"]["values"] = valores
dict_series["carga_marina"]["dates"] = carga_marina[:, :timestamp]

p1 = plot(dict_series["vazao"]["dates"],dict_series["vazao"]["values"],label="")
p2 = histogram(dict_series["vazao"]["values"], bins=25,label="")
plot(p1,p2, layout = (2,1) , plot_title="Série de Vazão e seu histograma")
savefig(current_path*"\\Saidas\\Relatorio\\SeriesTestes\\vazao.png")

" ----- GAS-CNO LogNormal ------ "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "carga_marina"
y = log.(dict_series[serie]["values"])
dates = dict_series[serie]["dates"]

y_norm = normalize_data(y)

steps_ahead = 12
len_train = length(y) - steps_ahead

y_train = y_norm[1:len_train]
y_test = y_norm[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "LogNormal"
dist = UnobservedComponentsGAS.NormalDistribution(missing, missing)


DICT_MODELS["LogNormal"] = Dict() 

DICT_MODELS["LogNormal"]["carga_marina"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 1.0, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, false, "multiplicative")

DICT_MODELS["LogNormal"]["vazao"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 1.0, Dict(2 => false, 1=>false),  
                                                            Dict(1 => true, 2=>false),  Dict(1 => false, 2 => false), 
                                                            Dict(1 => 12), false, false, "multiplicative")

DICT_MODELS["LogNormal"]["ena"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 1.0, Dict(1=>false), 
                                                            Dict(1=>false), Dict(1 => 2), 
                                                            Dict(1 => 12), false, false, "multiplicative")

num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]
fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train)

# gas_model = DICT_MODELS[distribution][serie]
# auto_gas_output = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead)
# fitted_model = auto_gas_output[1]

std_residuals = get_residuals(fitted_model, distribution, y_train, true)
residuals = get_residuals(fitted_model, distribution, y_train, false)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious)

fitted_model.fit_in_sample = denormalize_data(fitted_model.fit_in_sample, y)
y_train = denormalize_data(y_train, y)
y_test = denormalize_data(y_test, y)
forecast["mean"] = denormalize_data(forecast["mean"], y)
        
" ---- Visualizando os resíduos, fit in sample e forecast ----- "

recover_scale = true

recover_scale ? scale="Original" : scale="Log"

path_saida = current_path*"\\Saidas\\CombNaoLinear\\$distribution\\$scale\\"

dict_params = DataFrame(get_parameters(fitted_model))
CSV.write(path_saida*"$(serie)_params.csv",dict_params)

plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

(fitted_model, forecast, dates_train, y_train, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_fit_forecast_$(distribution).png")

plot_residuals(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

plot_acf_residuals(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

plot_residuals_histogram(std_residuals,distribution, serie)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_components_$(distribution).png")

plot_qqplot(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

plot_diagnosis(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

mapes = get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)


plot((components["level"].+components["slope"]).*components["seasonality"])


" -------------------- GAS-CNO Gamma -------------------- "

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")

serie = "ena"
y = dict_series[serie]["values"]
dates = dict_series[serie]["dates"]

y_norm = normalize_data(y)

steps_ahead = 12
len_train = length(y) - steps_ahead

y_train = y_norm[1:len_train]
y_test = y_norm[len_train+1:end]

dates_train = dates[1:len_train]
dates_test = dates[len_train+1:end]

distribution = "Gamma"
dist = UnobservedComponentsGAS.GammaDistribution(missing, missing)

DICT_MODELS[distribution] = Dict() 

DICT_MODELS[distribution]["carga"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 0.0, Dict(1=>false),  
                                                        Dict(1 => true),  Dict(1 => false), 
                                                        Dict(1 => 12), false, true)

DICT_MODELS[distribution]["vazao"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 1.0, Dict(2 => false, 1=>false),  
                                                            Dict(1 => true, 2=>false),  Dict(1 => false, 2 => false), 
                                                            Dict(1 => 12), false, false)

DICT_MODELS[distribution]["ena"]=UnobservedComponentsGAS.GASModel(dist, [true, false], 0.5, Dict(1=>false), 
                                                            Dict(1=>false), Dict(1 => 1), 
                                                            Dict(1 => 12), false, false)


num_scenarious = 500

gas_model = DICT_MODELS[distribution][serie]
fitted_model = UnobservedComponentsGAS.fit(gas_model, y_train)

# gas_model = UnobservedComponentsGAS.GASModel(dist, time_varying_params, missing, random_walk, random_walk_slope, ar, seasonality, robust, stochastic)
# auto_gas_output = UnobservedComponentsGAS.auto_gas(gas_model, y_train, steps_ahead)
# auto_gas_output
# fitted_model = auto_gas_output[1]

residuals = get_residuals(fitted_model, distribution, y_train, false)
std_residuals = get_residuals(fitted_model, distribution, y_train, true)
forecast = UnobservedComponentsGAS.predict(gas_model, fitted_model, y_train, steps_ahead, num_scenarious)

fitted_model.fit_in_sample = denormalize_data(fitted_model.fit_in_sample, y)
y_train = denormalize_data(y_train, y)
y_test = denormalize_data(y_test, y)
forecast["mean"] = denormalize_data(forecast["mean"], y)

" ---- Visualizando os resíduos, fit in sample e forecast ----- "
path_saida = current_path*"\\Saidas\\CombNaoLinear\\$distribution\\"
recover_scale = false

dict_params = DataFrame(get_parameters(fitted_model))
CSV.write(path_saida*"$(serie)_params.csv",dict_params)

plot_fit_in_sample(fitted_model, dates_train, y_train, distribution, recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_fit_in_sample_$(distribution).png")

plot_forecast(fitted_model, forecast, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_forecast_$(distribution).png")

(fitted_model, forecast, dates_train, y_train, y_test, dates_test, distribution, residuals, recover_scale, serie)
savefig(path_saida*"$(serie)_fit_forecast_$(distribution).png")

plot_residuals(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_residuals_$(distribution).png")

plot_acf_residuals(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_residuals_acf_$(distribution).png")

plot_residuals_histogram(std_residuals,distribution, serie)
savefig(path_saida*"$(serie)_residuals_histogram_$(distribution).png")

residuals_diagnostics_05 = get_residuals_diagnostics(residuals, 0.05, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_05.csv",residuals_diagnostics_05)

residuals_diagnostics_01 = get_residuals_diagnostics(residuals, 0.01, fitted_model)
CSV.write(path_saida*"$(serie)_residuals_diagnostics_01.csv",residuals_diagnostics_01)

plot_components(fitted_model, dates_train, distribution, "param_1", recover_scale, residuals, serie)
savefig(path_saida*"$(serie)_components_$(distribution).png")

plot_qqplot(std_residuals, distribution, serie)
savefig(path_saida*"$(serie)_qqplot_$(distribution).png")

plot_diagnosis(std_residuals, dates_train, distribution, true, serie)
savefig(path_saida*"$(serie)_diagnosticos_$(distribution).png")

mapes = get_mapes(y_train, y_test, fitted_model, forecast, residuals ,recover_scale)
CSV.write(path_saida*"$(serie)_mapes.csv",mapes)



" AutoARIMA Benchmark"

path_saida = current_path*"\\Saidas\\Benchmark\\AutoARIMA\\"

dict_benchmarks          = Dict()
dict_benchmarks["carga"] = Dict()
dict_benchmarks["ena"]   = Dict()
dict_benchmarks["vazao"] = Dict()

dict_benchmarks["carga"]["fit"]      = CSV.read(path_series*"carga_fit_autoarima.csv",DataFrame)
dict_benchmarks["carga"]["forecast"] = CSV.read(path_series*"carga_forecast_autoarima.csv",DataFrame)

dict_benchmarks["ena"]["fit"]      = CSV.read(path_series*"ena_fit_autoarima.csv",DataFrame)
dict_benchmarks["ena"]["forecast"] = CSV.read(path_series*"ena_forecast_autoarima.csv",DataFrame)

dict_benchmarks["vazao"]["fit"]      = CSV.read(path_series*"vazao_fit_autoarima.csv",DataFrame)
dict_benchmarks["vazao"]["forecast"] = CSV.read(path_series*"vazao_forecast_autoarima.csv",DataFrame)


for serie in ["carga", "ena", "vazao"]
    
    y = dict_series[serie]["values"]
    dates = dict_series[serie]["dates"]
    fit = dict_benchmarks[serie]["fit"].Values
    dates_fit = dict_benchmarks[serie]["fit"].Data
    prev = dict_benchmarks[serie]["forecast"].Values
    dates_prev = dict_benchmarks[serie]["forecast"].Data

    residuals = y[1:end-12] .- fit
    std_residuals = (residuals .- mean(residuals))./std(residuals)

    mape_train = round(100*MAPE(y[1:end-12],fit), digits=2)
    mape_test = round(100*MAPE(y[end-11:end],prev), digits=2)
    df_mapes = DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
    CSV.write(path_saida*"$(serie)_mapes.csv",df_mapes)

    plot(title="Fit in sample AutoARIMA $serie")
    plot!(dates, y, label="Série")
    plot!(dates_fit, fit, label="Fit: MAPE = $mape_train%")
    savefig(path_saida*"$(serie)_fit_autoarima.png")

    plot(title="Forecast AutoARIMA $serie")
    plot!(dates[end-11:end], y[end-11:end], label="Série")
    plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
    savefig(path_saida*"$(serie)_forecast_autoarima.png")

    plot(title="Fit and Forecast AutoARIMA $serie")
    plot!(dates, y, label="Série")
    plot!(dates_fit, fit, label="Fit; MAPE = $mape_train%")
    plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
    savefig(path_saida*"$(serie)_fit_forecast_autoarima.png")

    qq = plot(qqplot(Normal, std_residuals))
    qq = plot!(title="QQPlot Residuos")
    h = histogram(std_residuals, title="Histograma Residuos", label="")
    r = plot(title="Resíduos Padronizados")
    r = plot!(dates_fit, std_residuals , label="Resíduos")
    acf_values = autocor(std_residuals)
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval
    a = plot(title="FAC dos Residuos")
    a = plot!(autocor(std_residuals),seriestype=:stem, label="")
    a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
    plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
        plot_title = "Diagnosticos Residuos AutoARIMA - $serie", title=["Resíduos Padronizados" "FAC dos Residuos" "Histograma Residuos" "QQPlot Residuos"])
    savefig(path_saida*"$(serie)_diagnosticos_autoarima.png")    
end

