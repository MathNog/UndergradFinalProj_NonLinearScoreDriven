module FuncoesTeste

using CSV, DataFrames, Dates, Parameters, Plots, MLJBase, Statistics
using StatsBase, UnPack
using ARCHModels, HypothesisTests
using StatsPlots, Distributions, SpecialFunctions
using JuMP, Ipopt

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
        println(param)
        for (key,value) in fitted_model.components[param]
            println(" ", key,)
            if key != "intercept"
                for (key2, value2) in value["hyperparameters"]
                    println("   ", key2)
                    # println("   ", value2)
                    if (key2 !="γ") && (key2!="γ_star")
                        dict_params[param*"_"*key*"_"*key2] = value2
                    end
                end
            end
        end
    end
    return dict_params
end


function get_std_residuals(y, fit_in_sample, var)
    return (y .- fit_in_sample) ./ sqrt.(var)
end


# function get_residuals(fitted_model, model, y, standarize)

#     r = (y .- fitted_model.fit_in_sample)[2:end]
#     if standarize
#         return (r.-mean(r))./std(r)
#     else
#         return r
#     end
# end

function get_quantile_residuals(fitted_model)
    return fitted_model.residuals["q_residuals"]
end

function plot_residuals(residuals, dates, model, std_bool, serie, type, combination, d)
    
    if type == "quantile"
        plot(title="Residuos Quantílicos $model - $serie - $combination - d = $d", titlefontsize=12)
        plot!(dates, residuals, label="Resíduos Quantílicos")
    else
        std_bool==true ? std_title = "Padronizados" : std_title = ""
        plot(title="Resíduos $std_title $model - $serie - $combination - d = $d", titlefontsize=12)
        plot!(dates, residuals , label="Resíduos")
    end
end

function plot_acf_residuals(residuals, model, serie, type, combination, d)
    
    acf_values = autocor(residuals)[1:15]
    lag_values = collect(0:14)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval

    type == "quantile" ? tipo = "Quantílicos" : tipo = ""
    plot(title="FAC dos Residuos $tipo $model - $serie - $combination - d = $d", titlefontsize=11)
    plot!(acf_values,seriestype=:stem, label="")
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
    d = get_residuals_diagnosis_pvalues(residuals, fitted_model)
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

function plot_residuals_histogram(residuals, model, serie, type, combination, d)
    if type == "quantile"
        histogram(residuals, title="Histograma Residuos Quantílicos $model - $serie - $combination - d = $d", label="", titlefontsize=11)
    else
        histogram(residuals, title="Histograma Residuos $model - $serie - $combination - d = $d", label="", titlefontsize=11)
    end
end

function plot_fit_in_sample(fitted_model, fit_dates, y_train, model, recover_scale, residuals, serie, combination, d)
    
    fit_in_sample = fitted_model.fit_in_sample[2:end]
    # println(fit_in_sample)
    if recover_scale
        y_train = exp.(y_train)
        K = get_number_parameters(fitted_model)
        fit_in_sample = correct_scale(fit_in_sample, K, residuals)
    end
    # println(fit_in_sample)
    plot(fit_dates[2:end], y_train[2:end], label="Série")
    plot!(fit_dates[2:end], fit_in_sample, label="Fit in sample")    
    plot!(title=" Fit in sample GAS-CNO $model - $serie - $combination - d = $d", titlefontsize=11)
    
end

function plot_forecast(fitted_model, forecast, y_test, forecast_dates, model, residuals, recover_scale, serie, combination, d)

    if recover_scale
        y_test = exp.(y_test)
        K = get_number_parameters(fitted_model)
        forecast_mean = correct_scale(forecast["mean"], K, residuals)    
    else
        forecast_mean = forecast["mean"]
    end
    p = plot(title = "Forecast GAS-CNO $model - $serie - $combination - d = $d", titlefontsize=11)
    p = plot!(forecast_dates, y_test, label="Série")
    p = plot!(forecast_dates, forecast_mean, label="Forecast", color="red")
    display(p)
end

function plot_fit_forecast(fitted_model, forecast,fit_dates, y_train, y_test, forecast_dates, model, residuals, recover_scale, serie, combination, d)
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
    p = plot(title = "Fit and Forecast GAS-CNO $model - $serie - $combination - d = $d", titlefontsize=11)
    p = plot!(dates, y, label="Série")
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

function plot_components(fitted_model, estimation_dates, model, param, recover_scale, residuals, serie, combination, d)
    components = get_components(fitted_model, param, recover_scale, residuals)
    
    "level" in keys(components) ? level = components["level"] : level = ones(length(estimation_dates)).*missing
    "slope" in keys(components) ? slope = components["slope"] : slope = ones(length(estimation_dates)).*missing
    "seasonality" in keys(components) ? seasonality = components["seasonality"] : seasonality = ones(length(estimation_dates)).*missing
    "ar" in keys(components) ? ar = components["ar"] : ar = ones(length(estimation_dates)).*missing

    if "ar" in keys(components)
        p1 = plot(estimation_dates[2:end], ar[2:end], label="AR")
        p3 = plot(estimation_dates[2:end], seasonality[2:end], label="Seasonality")
        plot(p1, p3, layout = (2,1) ,plot_title = "Componentes GAS-CNO $model - $serie - $combination - d = $d", plot_titlefontsize=11)
    else
        p1 = plot(estimation_dates[2:end], level[2:end], label="Level")
        p2 = plot(estimation_dates[2:end],slope[2:end], label="Slope")
        p3 = plot(estimation_dates[2:end], seasonality[2:end], label="Seasonality")
        plot(p1, p2, p3, layout = (3,1) ,plot_title = "Componentes GAS-CNO $model - $serie - $combination - d = $d", plot_titlefontsize=11)
    end
end

function plot_qqplot(residuals, model, serie, type, combination, d)
    plot(qqplot(Normal, residuals))
    if type == "quantile"
        plot!(title="QQPlot Residuos Quantílicos $model - $serie - $combination - d = $d", titlefontsize=11)
    else
        plot!(title="QQPlot Residuos $model - $serie - $combination - d = $d", titlefontsize=11)
    end
end

function plot_diagnosis(residuals, dates, model, std_bool, serie, type, combination, d)
    
    @info "QQPlot"
    type== "quantile" ? tipo = " Quantílicos" : tipo = ""
    qq = plot(qqplot(Normal, residuals))
    qq = plot!(title="QQPlot Residuos$tipo")

    @info "Histograma"
    h = histogram(residuals, title="Histograma Residuos$tipo", label="")

    @info "Residuos"
    # std_bool==true ? res = (residuals.-mean(residuals))./std(residuals) : res = residuals
    std_bool==true ? std_title = "Padronizados" : std_title = ""
    r = plot(title="Resíduos $tipo$std_title")
    r = plot!(dates, residuals , label="Resíduos$tipo")

    @info "ACF"
    acf_values = autocor(residuals)
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval

    a = plot(title="FAC dos Residuos$tipo")
    a = plot!(collect(0:14), autocor(residuals)[1:15],seriestype=:stem, label="")
    a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")

    @info "Todos"
    plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
        plot_title = "Diagnosticos Residuos $tipo GAS-CNO $model - $serie - $combination - d = $d", title=["Resíduos $tipo$std_title" "FAC dos Residuos$tipo" "Histograma Residuos$tipo" "QQPlot Residuos$tipo"])
    
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

function scale_data(vec::Vector{T}, min_val::T, max_val::T) where T
    # Find the minimum and maximum values of the vector
    min_vec = minimum(vec)
    max_vec = maximum(vec)

    # Scale the vector to the desired range
    scaled_vec = min_val .+ (max_val - min_val) .* (vec .- min_vec) ./ (max_vec - min_vec)

    return scaled_vec
end

function unscale_data(scaled_vec::Vector{T}, original_vec::Vector{T}) where T
    # Find the minimum and maximum values of the original vector
    min_original = minimum(original_vec)
    max_original = maximum(original_vec)

    # Find the minimum and maximum values of the scaled vector
    min_scaled = minimum(scaled_vec)
    max_scaled = maximum(scaled_vec)

    # Unscaled the vector to the original range
    unscaled_vec = min_original .+ (max_original - min_original) .* (scaled_vec .- min_scaled) ./ (max_scaled - min_scaled)

    return unscaled_vec
end

function unscale_data(scenarios::Matrix{T}, original_vec::Vector{T}) where{T}
    scenarios_orig = deepcopy(scenarios)
    for s in 1:size(scenarios, 2)
        scenarios_orig[:,s] = unscale_data(scenarios[:,s], original_vec)
    end
    return scenarios_orig
end

function denormalize_data(scenarios::Matrix{Fl}, y::Vector{Fl}) where{Fl}
    scenarios_orig = deepcopy(scenarios)
    for s in 1:size(scenarios, 2)
        scenarios_orig[:,s] = denormalize_data(scenarios[:,s], y)
    end
    return scenarios_orig
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


function plot_forecast_histograms(fitted_model, forecast, residuals, model, serie, bins, recover_scale, combination, d)

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
        plot_title = "Histogramas dos cenários de previsão - $model - $serie - $combination - d = $d")
    
end

function fit_harmonics(y::Vector{Fl}, seasonal_period::Int64, stochastic::Bool) where {Fl}

    T = length(y)

    if seasonal_period % 2 == 0
        num_harmonic = Int64(seasonal_period / 2)
    else
        num_harmonic = Int64((seasonal_period -1) / 2)
    end

    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    set_optimizer_attribute(model, "print_level", 0)
    #set_optimizer_attribute(model, "hessian_constant", "yes")

    @variable(model, y_hat[1:T])

    if stochastic
        @variable(model, γ[1:num_harmonic, 1:T])
        @variable(model, γ_star[1:num_harmonic, 1:T])

        @constraint(model, [i = 1:num_harmonic, t = 2:T], γ[i, t] == γ[i, t-1] * cos(2*π*i / seasonal_period) + 
                                                                    γ_star[i,t-1]*sin(2*π*i / seasonal_period))
        @constraint(model, [i = 1:num_harmonic, t = 2:T], γ_star[i, t] == -γ[i, t-1] * sin(2*π*i / seasonal_period) + 
                                                                                γ_star[i,t-1]*cos(2*π*i / seasonal_period))

        @constraint(model, [t = 1:T], y_hat[t] == sum(γ[i, t] for i in 1:num_harmonic))
    else

        @variable(model, γ[1:num_harmonic])
        @variable(model, γ_star[1:num_harmonic])

        @constraint(model, [t = 1:T], y_hat[t] == sum(γ[i] * cos(2 * π * i * t/seasonal_period) + 
                                                  γ_star[i] * sin(2 * π * i* t/seasonal_period)  for i in 1:num_harmonic))
    end
   
    @objective(model, Min, sum((y .- y_hat).^2))
    optimize!(model)

    return JuMP.value.(γ), JuMP.value.(γ_star)
    
end


function get_initial_α(y::Vector{Float64})

    T = length(y)
    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, α >= 1e-2)  # Ensure α is positive
    @variable(model, λ[1:T] .>= 1e-4)
    register(model, :Γ, 1, Γ; autodiff = true)
    @NLobjective(model, Max, sum(-log(Γ(α)) - α*log(1/α) - α*log(λ[i]) +(α-1)*log(y[i]) - (α/λ[i])*y[i] for i in 1:T))
    optimize!(model)
    if has_values(model)
        return JuMP.value.(α)
    else
        return fit_mle(Gamma, y).α
    end 
end


function get_initial_params_normal(y::Vector{Fl}, time_varying_params::Vector{Bool}) where Fl

    T         = length(y)
    # dist_code = get_dist_code(dist)
    seasonal_period = 12#get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist) #(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = var(diff(y))#get_initial_σ(y)^2#
    end

    return initial_params
end


"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"
function get_initial_params_gamma(y::Vector{Fl}, time_varying_params::Vector{Bool}) where Fl

    println("Inicialização dos parâmetros iniciais")
    T         = length(y)
    # dist_code = get_dist_code(dist)
    seasonal_period = 12#get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()
    fitted_distribution = fit_mle(Gamma, y)
    
    # param[2] = λ = média
    if time_varying_params[2]
        println("λ = y")
        initial_params[2] = y
    else
        println("λ = mean(y)")
        initial_params[2] = fitted_distribution.α*fitted_distribution.θ
    end

    # param[1] = α
    if time_varying_params[1]
        println("α = ??")
        initial_params[1] = get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        println("α = $(fitted_distribution.α)")
        println(("α = $(get_initial_α(y))"))
        initial_params[1] = get_initial_α(y)#mean(y)^2/var((y)) 
    end
    
    return initial_params
end
 
 

function get_initial_values_from_components(y, components, stochastic, serie, dist)
    
    T = size(components, 1)
    if serie == "carga"
        initial_rws         = components[:,"l"]
        initial_slope       = components[:,"b"]
        initial_seasonality = components[:,"s1"]
        initial_ar          = zeros(T)
    else
        initial_rws         = zeros(T)
        initial_slope       = zeros(T)
        initial_ar          = components[:,"l"]
        initial_seasonality = components[:,"s1"]
    end

    initial_values                          = Dict{String,Any}()
    if dist == "Gamma" 
        initial_params = get_initial_params_gamma(y, [false, true])
        initial_values["fixed_param"]           = [initial_params[1]]
        initial_values["param"]                 = initial_params[2]
    else
        initial_params = get_initial_params_normal(y, [true, false])
        initial_values["fixed_param"]           = [initial_params[2]]
        initial_values["param"]                 = initial_params[1]
    end

    initial_γ, initial_γ_star = fit_harmonics(initial_seasonality, 12, stochastic)

   

    

    initial_values["intercept"]             = Dict()
    initial_values["intercept"]["values"]   = [0.02]
    initial_values["rw"]                    = Dict()
    initial_values["rw"]["values"]          = zeros(T)
    initial_values["rw"]["κ"]               = 0.02
    initial_values["rws"]                   = Dict()
    initial_values["rws"]["values"]         = initial_rws
    initial_values["rws"]["κ"]              = 0.02
    initial_values["slope"]                 = Dict()
    initial_values["slope"]["values"]       = initial_slope
    initial_values["slope"]["κ"]            = 0.02
    initial_values["seasonality"]           = Dict()
    initial_values["seasonality"]["values"] = initial_seasonality
    initial_values["seasonality"]["γ"]      = initial_γ
    initial_values["seasonality"]["γ_star"] = initial_γ_star
    initial_values["seasonality"]["κ"]      = 0.02
    initial_values["ar"]                    = Dict()
    initial_values["ar"]["values"]          = initial_ar
    initial_values["ar"]["ϕ"]               = [0.]
    initial_values["ar"]["κ"]               = 0.02
    return initial_values
end


end