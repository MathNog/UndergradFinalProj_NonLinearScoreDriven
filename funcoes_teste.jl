module FuncoesTeste

using CSV, DataFrames, Dates, Parameters, Plots, MLJBase, Statistics
using StatsBase, UnPack
using ARCHModels, HypothesisTests
using StatsPlots, Distributions, SpecialFunctions

function MAPE(A, F)
    return MLJBase.mape(F,A)
end

function Γ(x)
    return SpecialFunctions.gamma(x)
end

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

function plot_fit_forecast(fitted_model, forecast,fit_dates, y_train, y_test, forecast_dates, model, residuals, recover_scale, serie)
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
    return ((y .- min) ./ (max - min)) .+ 0.5
end

function denormalize_data(y_norm::Vector{Fl}, y::Vector{Fl}) where{Fl}
    min = minimum(y)
    max = maximum(y)
    return (y_norm .- 0.5) .* (max-min) .+ min
end

function get_forecast_quantiles(forecast, steps)
    df_forecast_quantiles = DataFrame(zeros(3,6),:auto)
    rename!(df_forecast_quantiles, ["Steps","Q5%","Q25%","Q50%","Q75%","Q95%"])
    for (i,step) in enumerate(steps)
        q = quantile!(forecast["scenarios"][i,:],[0.05,0.25,0.5,0.75,0.95])
        df_forecast_quantiles[i,:] = vcat(Int64(step),q)
    end
    return df_forecast_quantiles
end


function plot_forecast_histograms(fitted_model, forecast, residuals, model, serie, bins, recover_scale)

    if recover_scale
        K = get_number_parameters(fitted_model)
        for i in 1:size(forecast["scenarios"],2)
            forecast["scenarios"][:,i] = correct_scale(forecast["scenarios"][:,i], K, residuals)  
        end
    end
    h1 = histogram(forecast["scenarios"][1,:], title="Histograma Previsão 1 Passo à frente", label="", bins=bins)
    h5 = histogram(forecast["scenarios"][5,:], title="Histograma Previsão 5 Passos à frente", label="", bins=bins)
    h12 = histogram(forecast["scenarios"][12,:], title="Histograma Previsão 12 Passos à frente", label="", bins=bins)

    plot(h1, h5, h12,  layout=grid(3,1), size=(900,800), 
        plot_title = "Histogramas dos cenários de previsão - $model - $serie")
    
end
end