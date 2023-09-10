using CSV
using Statistics
using DataFrames
using Dates
using TimeSeriesInterface
using Plots
using StatsBase, UnPack
using ARCHModels, HypothesisTests

import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("UnobservedComponentsGAS/src/UnobservedComponentsGAS.jl")
include("src/EnergisaProjetoDeMercadoForecastingModels.jl")

function MAPE(A, F)
    return 100*mean(abs.((A .- F)./F))
end


" ------- Criando dicionario de Dados ------- "

path_series = "C:/Users/matno/OneDrive/Documentos/PUC/0_Períodos/TCC/TCC/Dados/Tratados/"

ena = CSV.read(path_series*"ena_limpo.csv",DataFrame)
vazao = CSV.read(path_series*"vazao_limpo.csv", DataFrame)
carga = CSV.read(path_series*"carga_limpo.csv", DataFrame)

dict_series = Dict()
dict_series["ena"] = Dict()
dict_series["vazao"] = Dict()
dict_series["carga"] = Dict()

dict_series["ena"]["values"] = ena[:,:ENA]
dict_series["ena"]["dates"] = ena[:,:Data]
dict_series["vazao"]["values"] = parse.(Float64,vazao[:,:Vazao])
dict_series["vazao"]["dates"] = vazao[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"] = carga[:,:Data]


plot(dict_series["carga"]["dates"],dict_series["carga"]["values"])



" ------------- Funções Auxiliares --------------"

function get_correct_forecast(output)
    residuals = output.residuals["not_treated"].vals
    forecast_log = output.forecast["conditional"]["mean"].forecast

    SQR = sum(residuals.^2)
    N = length(residuals)
    K = 16
    σ² = SQR/(N-K)
    return exp.(forecast_log)*exp(0.5*σ²)
end

function get_correct_fit(output)
    residuals = output.residuals["not_treated"].vals
    fit_in_sample_log = output.fit_in_sample.vals

    SQR = sum(residuals.^2)
    N = length(residuals)
    K = 16
    σ² = SQR/(N-K)
    return exp.(fit_in_sample_log)*exp(0.5*σ²)
end

function correct_scale(component, output, K)
    residuals = output.residuals["not_treated"].vals

    SQR = sum(residuals.^2)
    N = length(residuals)
    σ² = SQR/(N-K)
    return exp.(component)*exp(0.5*σ²)
end

function get_number_parameters(output)
    return length(output.hyperparameters["model"])
end

function plot_residuals(output, standarize, model)
    res = output.residuals["not_treated"].vals
    dates = output.residuals["not_treated"].timestamps
    if standarize 
        res = (res .- mean(res)) ./ std(res)
        label = "Residuos"
    else
        label = "Residuos Padronizados"
    end
    plot(title="$label $model")
    plot!(dates, res , label=label)
end

function plot_acf_residuals(output, model)
    residuals = output.residuals["not_treated"].vals
    acf_values = autocor(residuals)
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(y_train))  # 95% confidence interval

    plot(title="FAC dos Residuos $model")
    plot!(autocor(residuals),seriestype=:stem, label="")
    hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
end


function get_residuals_diagnosis(output)
    dof = get_number_parameters(output)
    residuals = output.residuals["not_treated"].vals
    # Jarque Bera
    jb = pvalue(JarqueBeraTest(residuals))
    # Ljung Box
    lb = pvalue(LjungBoxTest(residuals, 24, dof))
    # ARCH
    arch = pvalue(ARCHLMTest(residuals, 12))
    #ADF
    adf = pvalue(ADFTest(residuals, :none, 1))
    
    return Dict(:lb=>lb, :jb=>jb, :adf=>adf, :arch=>arch)
end

function print_residuals_diagnostics(output, α)
    d = get_residuals_diagnosis(output)
    nomes = Dict(:lb=>"Ljung Box", :jb=>"Jarque Bera", :adf=>"ADF", :arch=>"ARCHLM")
    for key in keys(d)
        nome = nomes[key]
        if d[key] < α
            println("Teste $nome rejeita H₀ a um nível de $α")
        else
            println("Teste $nome não rejeita H₀ a um nível de $α")
        end
    end
end

function plot_residuals_histogram(output, model)
    residuals = output.residuals["not_treated"].vals
    histogram(residuals, title="Histograma Residuos $model", label="")
end

function plot_fit_in_sample(output, model, recover_scale)
    y = output.fitted_model["data_parameters"].dependent_series.vals
    estimation_dates = Date.(output.fitted_model["data_parameters"].dependent_series.timestamps)
    fit_dates = Date.(output.fit_in_sample.timestamps)
    
    if recover_scale
        y = exp.(y)
        fit_in_sample = get_correct_fit(output)
    else
        fit_in_sample = output.fit_in_sample.vals
    end

    plot(estimation_dates, y, label="Série")
    plot!(fit_dates, fit_in_sample, label="Fit in sample")    
    plot!(title=" Fit in sample GAS-CNO $model")
    
end

function plot_forecast(output, y_test, model, recover_scale)
    forecast = output.forecast["conditional"]["mean"].forecast
    forecast_dates = Date.(output.forecast["conditional"]["mean"].timestamps)

    if recover_scale
        y_test = exp.(y_test)
        forecast = get_correct_forecast(output)    
    end
    p = plot(title = "Forecast $model")
    p = plot!(forecast_dates, y_test, label="Série")
    p = plot!(forecast_dates, forecast, label="Forecast", color="red")
    display(p)
end

function get_components(output, param, recover_scale)    
    dict_components = output.fitted_model["model"]["fitted_gas"].components[param]
    dof = get_number_parameters(output)
    if recover_scale
        components = Dict(
            "slope" => correct_scale(dict_components["slope"]["value"], output, dof),
            "level" => correct_scale(dict_components["level"]["value"], output, dof),
            "seasonality" => correct_scale(dict_components["seasonality"]["value"], output, dof),
        )
    else     
        components = Dict(
            "slope" => dict_components["slope"]["value"],
            "level" => dict_components["level"]["value"],
            "seasonality" => dict_components["seasonality"]["value"],
        )
    end
    return components
end


function plot_components(output, model, param, recover_scale)
    @unpack slope, level, seasonality = get_components(output, param, recover_scale)
    estimation_dates = Date.(output.fitted_model["data_parameters"].dependent_series.timestamps)
    
    p1 = plot(estimation_dates[2:end], level[2:end], label="Level")
    p2 = plot(estimation_dates[2:end],slope[2:end], label="Slope")
    p3 = plot(estimation_dates[2:end], seasonality[2:end], label="Seasonality")
    plot(p1, p2, p3, layout = (3,1) ,plot_title = "Componentes GAS-CNO $model $param")
end

" ----- GAS-CNO Normal - Usando código da ferramenta de mercado ----- "

distribution = "Normal"
d = 0.5
random_walk = Dict(1=>false,2=>false)
random_walk_slope = Dict(1=>true,2=>false)
seasonality = 12
regularization_factor = 0.5
robust = false

y = dict_series["carga"]["values"]
dates = dict_series["carga"]["dates"]

# plot(dates, y)

len_test = 12
len_train = length(y) - len_test
y_train = y[1:len_train]
y_test = y[len_train+1:end]

estimation_dates = dates[1:len_train]
forecast_dates = dates[len_train+1:end]

dependent_series = TimeSeries("dependente", estimation_dates, y_train)

granularity = EnergisaProjetoDeMercadoForecastingModels.DataModel.Granularity(4)
data = EnergisaProjetoDeMercadoForecastingModels.DataModel.DataParameters(granularity, forecast_dates, dependent_series, [])

input = EnergisaProjetoDeMercadoForecastingModels.GASModel.GASInput(distribution, d, random_walk, random_walk_slope,
                                                                    seasonality, regularization_factor, 1)

output = EnergisaProjetoDeMercadoForecastingModels.GASModel.fit_and_forecast(data, input; backtest = false, automatic_anomaly = false, run_simulate = false);

" ---- Visualizando os resíduos, fit in sample e forecast ----- "
path_saida = "C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\TCC\\TCC\\Saidas\\Benchmark\\Normal\\"

recover_scale = false
plot_fit_in_sample(output, "Normal", recover_scale)
savefig(path_saida*"fit_in_sample_Normal_carga.png")

plot_forecast(output, y_test, "Normal", recover_scale)
savefig(path_saida*"forecast_Normal_carga.png")

plot_residuals(output, true, "Normal")
savefig(path_saida*"residuals_Normal_carga.png")

plot_acf_residuals(output, "Normal")
savefig(path_saida*"residuals_acf_Normal_carga.png")

plot_residuals_histogram(output,"Normal")
savefig(path_saida*"residuals_histogram_Normal_carga.png")

residuals_diagnosis = get_residuals_diagnosis(output)

print_residuals_diagnostics(output, 0.05)
print_residuals_diagnostics(output, 0.01)

plot_components(output, "Normal", "param_1")
savefig(path_saida*"components_Normal_carga.png")



" ----- GAS-CNO LogNormal - Usando código da ferramenta de mercado ----- "

distribution = "Normal"
d = 0.5
random_walk = Dict(1=>false,2=>false)
random_walk_slope = Dict(1=>true,2=>false)
seasonality = 12
regularization_factor = 0.5
robust = false

y = log.(dict_series["carga"]["values"])
dates = dict_series["carga"]["dates"]

# plot(dates, y)

len_test = 12
len_train = length(y) - len_test
y_train = y[1:len_train]
y_test = y[len_train+1:end]

estimation_dates = dates[1:len_train]
forecast_dates = dates[len_train+1:end]

dependent_series = TimeSeries("dependente", estimation_dates, y_train)

granularity = EnergisaProjetoDeMercadoForecastingModels.DataModel.Granularity(4)
data = EnergisaProjetoDeMercadoForecastingModels.DataModel.DataParameters(granularity, forecast_dates, dependent_series, [])

input = EnergisaProjetoDeMercadoForecastingModels.GASModel.GASInput(distribution, d, random_walk, random_walk_slope,
                                                                    seasonality, regularization_factor, 1)

output = EnergisaProjetoDeMercadoForecastingModels.GASModel.fit_and_forecast(data, input; backtest = false, automatic_anomaly = false, run_simulate = false);


" ---- Pegando os resíduos, fit in sample e forecast ----- "
path_saida = "C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\TCC\\TCC\\Saidas\\Benchmark\\LogNormal\\"

recover_scale = true
recover_scale ? scale = "_original" : scale = "_log"

plot_fit_in_sample(output, "LogNormal", recover_scale)
savefig(path_saida*"fit_in_sample_LogNormal_carga$scale.png")

plot_forecast(output, y_test, "LogNormal", recover_scale)
savefig(path_saida*"forecast_LogNormal_carga$scale.png")

plot_residuals(output, true, "LogNormal")
savefig(path_saida*"residuals_LogNormal_carga$scale.png")

plot_acf_residuals(output, "LogNormal")
savefig(path_saida*"residuals_acf_LogNormal_carga$scale.png")

plot_residuals_histogram(output,"LogNormal")
savefig(path_saida*"residuals_histogram_LogNormal_carga$scale.png")

residuals_diagnosis = get_residuals_diagnosis(output)

print_residuals_diagnostics(output, 0.05)
print_residuals_diagnostics(output, 0.01)

plot_components(output, "LogNormal", "param_1", recover_scale)
savefig(path_saida*"components_LogNormal_carga$scale.png")