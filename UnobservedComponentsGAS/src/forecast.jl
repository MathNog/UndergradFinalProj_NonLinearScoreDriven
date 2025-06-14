"
Returns a dictionary with the fitted hyperparameters and components, with null forecast.
The components forecasts will be filled in the function predict_scenarios.
"
function get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, steps_ahead::Int64, num_scenarios::Int64; combination::String="additive")

    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic = gas_model

    idx_params = get_idxs_time_varying_params(time_varying_params) 
    order      = get_AR_order(ar)
    num_params = get_num_params(dist)
    components = output.components

    num_harmonic, seasonal_period = UnobservedComponentsGAS.get_num_harmonic_and_seasonal_period(seasonality)

    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    elseif length(idx_params) > length(num_harmonic) #considera os mesmos harmonicos para todos os parametros variantes, para não quebrar a update_S!
        num_harmonic = Int64.(ones(length(idx_params)) * num_harmonic[1])
    end
    
    T_fitted = length(output.fitted_params["param_1"])
    DICT_ZEROS_ONES= Dict("additive"=>zeros, "multiplicative1"=>ones, "multiplicative2"=>zeros, "multiplicative3"=>ones)


    dict_hyperparams_and_fitted_components                = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["rw"]          = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["rws"]         = Dict{String, Any}()
    dict_hyperparams_and_fitted_components["seasonality"] = Dict{String, Any}()    
    dict_hyperparams_and_fitted_components["ar"]          = Dict{String, Any}()

    dict_hyperparams_and_fitted_components["params"]       = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["intercept"]    = zeros(num_params)
    dict_hyperparams_and_fitted_components["score"]        = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["b_mult"]       = zeros(num_params)
  
    dict_hyperparams_and_fitted_components["rw"]["value"]  = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rw"]["κ"]      = zeros(num_params)

    dict_hyperparams_and_fitted_components["rws"]["value"] = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rws"]["b"]     = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["rws"]["κ"]     = zeros(num_params)
    dict_hyperparams_and_fitted_components["rws"]["κ_b"]   = zeros(num_params)
    dict_hyperparams_and_fitted_components["rws"]["ϕb"]     = zeros(num_params)

    dict_hyperparams_and_fitted_components["seasonality"]["value"]   = DICT_ZEROS_ONES[combination](num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["seasonality"]["κ"]       = DICT_ZEROS_ONES[combination](num_params)

    if stochastic
        dict_hyperparams_and_fitted_components["seasonality"]["γ"]       = DICT_ZEROS_ONES[combination](num_harmonic[idx_params[1]], T_fitted + steps_ahead, num_params, num_scenarios) 
        dict_hyperparams_and_fitted_components["seasonality"]["γ_star"]  = DICT_ZEROS_ONES[combination](num_harmonic[idx_params[1]], T_fitted + steps_ahead, num_params, num_scenarios)
    else
        dict_hyperparams_and_fitted_components["seasonality"]["γ"]       = DICT_ZEROS_ONES[combination](num_harmonic[idx_params[1]], num_params) 
        dict_hyperparams_and_fitted_components["seasonality"]["γ_star"]  = DICT_ZEROS_ONES[combination](num_harmonic[idx_params[1]], num_params)
    end
    
    dict_hyperparams_and_fitted_components["ar"]["value"]  = zeros(num_params, T_fitted + steps_ahead, num_scenarios)
    dict_hyperparams_and_fitted_components["ar"]["ϕ"]      = zeros(maximum(vcat(order...)), num_params)
    dict_hyperparams_and_fitted_components["ar"]["κ"]      = zeros(num_params)

    for i in 1:num_params
        dict_hyperparams_and_fitted_components["params"][i, 1:T_fitted, :] .= output.fitted_params["param_$i"]

        if i in idx_params
            dict_hyperparams_and_fitted_components["intercept"][i] = components["param_$i"]["intercept"]
            println("b_mult entrando")
            println(dict_hyperparams_and_fitted_components["b_mult"][i])
            dict_hyperparams_and_fitted_components["b_mult"][i]    = components["param_$i"]["b_mult"]
            println(dict_hyperparams_and_fitted_components["b_mult"][i])
        end

        if has_random_walk(random_walk, i)
            dict_hyperparams_and_fitted_components["rw"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["level"]["value"]
            dict_hyperparams_and_fitted_components["rw"]["κ"][i]                     = components["param_$i"]["level"]["hyperparameters"]["κ"]
        end

        if has_random_walk_slope(random_walk_slope, i)
            dict_hyperparams_and_fitted_components["rws"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["level"]["value"]
            dict_hyperparams_and_fitted_components["rws"]["κ"][i]                     = components["param_$i"]["level"]["hyperparameters"]["κ"]
            dict_hyperparams_and_fitted_components["rws"]["b"][i, 1:T_fitted, :]     .= components["param_$i"]["slope"]["value"]
            dict_hyperparams_and_fitted_components["rws"]["κ_b"][i]                   = components["param_$i"]["slope"]["hyperparameters"]["κ"]
            dict_hyperparams_and_fitted_components["rws"]["ϕb"][i]                     = components["param_$i"]["slope"]["hyperparameters"]["ϕb"]
        end

        if has_AR(ar, i)
            dict_hyperparams_and_fitted_components["ar"]["value"][i, 1:T_fitted, :] .= components["param_$i"]["ar"]["value"]
            dict_hyperparams_and_fitted_components["ar"]["ϕ"][:, i]                  = components["param_$i"]["ar"]["hyperparameters"]["ϕ"]
            dict_hyperparams_and_fitted_components["ar"]["κ"][i]                     = components["param_$i"]["ar"]["hyperparameters"]["κ"]
        end 

        if has_seasonality(seasonality, i)
            dict_hyperparams_and_fitted_components["seasonality"]["value"][i, 1:T_fitted, :]    .= components["param_$i"]["seasonality"]["value"]
            # dict_hyperparams_and_fitted_components["seasonality"]["κ"][i]                        = components["param_$i"]["seasonality"]["hyperparameters"]["κ"]
            if stochastic
                dict_hyperparams_and_fitted_components["seasonality"]["κ"][i]                        = components["param_$i"]["seasonality"]["hyperparameters"]["κ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, 1:T_fitted, i, :]     .= components["param_$i"]["seasonality"]["hyperparameters"]["γ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][:, 1:T_fitted, i,:] .= components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"]
            else
                # println(dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, i])
                # println(components["param_$i"]["seasonality"]["hyperparameters"]["γ"])
                dict_hyperparams_and_fitted_components["seasonality"]["γ"][:, i]       .= components["param_$i"]["seasonality"]["hyperparameters"]["γ"]
                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][:, i,] .= components["param_$i"]["seasonality"]["hyperparameters"]["γ_star"]
            end

        end
    end

    if length(idx_params) != num_params
        for i in setdiff(1:num_params, idx_params)
            dict_hyperparams_and_fitted_components["params"][i, :, :] .= output.fitted_params["param_$i"][1] #### PQ O [1] NO FINAL?
        end
    end

    return dict_hyperparams_and_fitted_components
end

"
Returns a dictionary with the fitted hyperparameters and components, with null forecast. Used when the model considers explanatory variables.
The components forecasts will be filled in the function predict_scenarios.
"
function get_dict_hyperparams_and_fitted_components_with_forecast(gas_model::GASModel, output::Output, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64; combination::String="string") where {Fl}
    
    n_exp      = size(X_forecast, 2)
    num_params = get_num_params(gas_model.dist)
    components = output.components

    dict_hyperparams_and_fitted_components = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, output, steps_ahead, num_scenarios;combination=combination)
    dict_hyperparams_and_fitted_components["explanatories"] = zeros(n_exp, num_params)

    dict_hyperparams_and_fitted_components["explanatories"][:, 1] = components["param_1"]["explanatories"]
    
    return dict_hyperparams_and_fitted_components

end

"
Updates the dict_hyperparams_and_fitted_components with the score scenario, for a specific time t.
"
function update_score!(dict_hyperparams_and_fitted_components::Dict{String, Any}, pred_y::Matrix{Float64}, d::Float64, dist_code::Int64, param::Int64, t::Int64, s::Int64)

    if size(dict_hyperparams_and_fitted_components["params"])[1] == 2
        dict_hyperparams_and_fitted_components["score"][param, t, s] = scaled_score(dict_hyperparams_and_fitted_components["params"][1, t - 1, s], 
                                                                                dict_hyperparams_and_fitted_components["params"][2, t - 1, s], 
                                                                                pred_y[t - 1, s], d, dist_code, param)
    elseif size(dict_hyperparams_and_fitted_components["params"])[1] == 3
        dict_hyperparams_and_fitted_components["score"][param, t, s] = scaled_score(dict_hyperparams_and_fitted_components["params"][1, t - 1, s], 
                                                                                dict_hyperparams_and_fitted_components["params"][2, t - 1, s],
                                                                                dict_hyperparams_and_fitted_components["params"][3, t - 1, s], 
                                                                                pred_y[t - 1, s], d, dist_code, param)
    end
    # println("Score = ", dict_hyperparams_and_fitted_components["score"][param, t, s])
end

"
Updates the dict_hyperparams_and_fitted_components with the random walk scenario, for a specific time t.
"
function update_rw!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)

    dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] = dict_hyperparams_and_fitted_components["rw"]["value"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rw"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
    # println("RW = ", dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s]) 
end

"
Updates the dict_hyperparams_and_fitted_components with the random walk and slope scenarios, for a specific time t.
"
function update_rws!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64)
    
    dict_hyperparams_and_fitted_components["rws"]["b"][param, t, s] = dict_hyperparams_and_fitted_components["rws"]["ϕb"][param]*dict_hyperparams_and_fitted_components["rws"]["b"][param, t - 1, s] + 
                                                                                    dict_hyperparams_and_fitted_components["rws"]["κ_b"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]

    dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] = dict_hyperparams_and_fitted_components["rws"]["value"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rws"]["b"][param, t - 1, s] + 
                                                                                        dict_hyperparams_and_fitted_components["rws"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
    # println("RWS = ", dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s])
end

"
Updates the dict_hyperparams_and_fitted_components with the seasonality scenarios, for a specific time t.
"
function update_S!(dict_hyperparams_and_fitted_components::Dict{String, Any}, num_harmonic::Vector{Int64}, diff_T::Int64, param::Int64, t::Int64, s::Int64)

    if length(size( dict_hyperparams_and_fitted_components["seasonality"]["γ"])) == 4
        for j in 1:num_harmonic[param]
            dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t, param, s] = dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t-1, param, s]*cos(2 * π * j /(num_harmonic[param] * 2)) +
                                                                                            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, t-1, param, 1]*sin(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                            dict_hyperparams_and_fitted_components["seasonality"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]

            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, t, param, s] = -dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t-1,param, s]*sin(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                                dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j,t-1, param, s]*cos(2 * π * j/(num_harmonic[param] * 2)) +
                                                                                                dict_hyperparams_and_fitted_components["seasonality"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
        end

        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, t, param, s] for j in 1:num_harmonic[param])
    else

        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["seasonality"]["γ"][j, param]*cos(2 * π * j * (diff_T + t)/(num_harmonic[param] * 2)) +
                                                                            dict_hyperparams_and_fitted_components["seasonality"]["γ_star"][j, param]*sin(2 * π * j * (diff_T + t)/(num_harmonic[param] * 2)) for j in 1:num_harmonic[param])
    end
    # println("Sazo = ",dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s])
end

"
Updates the dict_hyperparams_and_fitted_components with the autoregressive scenarios, for a specific time t.
"
function update_AR!(dict_hyperparams_and_fitted_components::Dict{String, Any}, order::Vector{Vector{Int64}} , param::Int64, t::Int64, s::Int64)

    dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] = sum(dict_hyperparams_and_fitted_components["ar"]["ϕ"][:, param][p] * dict_hyperparams_and_fitted_components["ar"]["value"][param, t - p, s] for p in order[param]) + 
                                                                                dict_hyperparams_and_fitted_components["ar"]["κ"][param] * dict_hyperparams_and_fitted_components["score"][param, t, s]
    # println("AR = ", dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s])
end

"
Updates the dict_hyperparams_and_fitted_components with the distribution parameters scenarios, for a specific time t.
"
function update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, param::Int64, t::Int64, s::Int64, dist::ScoreDrivenDistribution; combination::String="additive")

    m = dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] +
        dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] +
        dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s]
    b_mult = dict_hyperparams_and_fitted_components["b_mult"][param]
    
    # Colcoar link aqui nesses ifs
    if combination == "additive"
        # println("Combination $combination")
        # μ_t = m_t + s_t
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + m + 
                                                                        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s]
    elseif combination == "multiplicative1"
        # println("Combination $combination")
        # μ_t = m_t × s_t
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                    (m * dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s])
    elseif combination == "multiplicative2"
        # println("Combination $combination")
        # μ_t = m_t × (1 + s_t)
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                        (m * (1 .+ dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s]))
    else
        # println("Combination $combination")
        # μ_t = m_t + exp(b*m_t) × s_t
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                        (m + exp(b_mult*m) * dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s])
    end 
    
    # if typeof(dist) == UnobservedComponentsGAS.GammaDistribution
    #     println("Colocando funcao de ligação log/exp")
    #     dict_hyperparams_and_fitted_components["params"][param, t, s] = exp.(dict_hyperparams_and_fitted_components["params"][param, t, s])
    # end
                                                                                
end

"
Updates the dict_hyperparams_and_fitted_components with the distribution parameters scenarios, for a specific time t, with exogenous variables.
"
function update_params!(dict_hyperparams_and_fitted_components::Dict{String, Any}, X_forecast::Matrix{Fl}, period_X::Int64, param::Int64, t::Int64, s::Int64;combination::String="additive") where Fl

    n_exp = size(X_forecast, 2)
    println("Não é para entrar aqui")
    if combination=="additive"
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                        dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] + 
                                                                        dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] +
                                                                        dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] +
                                                                        dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] +
                                                                        sum(dict_hyperparams_and_fitted_components["explanatories"][j] * X_forecast[period_X, j] for j in 1:n_exp)
    else
        dict_hyperparams_and_fitted_components["params"][param, t, s] = dict_hyperparams_and_fitted_components["intercept"][param] + 
                                                                    (dict_hyperparams_and_fitted_components["rw"]["value"][param, t, s] * 
                                                                    dict_hyperparams_and_fitted_components["rws"]["value"][param, t, s] *
                                                                    dict_hyperparams_and_fitted_components["seasonality"]["value"][param, t, s] *
                                                                    dict_hyperparams_and_fitted_components["ar"]["value"][param, t, s] *
                                                                    sum(dict_hyperparams_and_fitted_components["explanatories"][j] * X_forecast[period_X, j] for j in 1:n_exp))
    end
end

"
Simulates scenarios considering the uncertaintity in the dynamics.
"
function simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64; combination::String="additive")
    
    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic, combination = gas_model

    idx_params      = get_idxs_time_varying_params(time_varying_params) 
    order           = get_AR_order(ar)
    num_harmonic, _ = get_num_harmonic_and_seasonal_period(seasonality)
    dist_code       = get_dist_code(dist)

    T        = length(y)
    T_fitted = length(output.fit_in_sample)

    first_idx = T - T_fitted + 1

    ### PORQUE RECALCULAR O NUM_HARMONIC?
   
    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    else
        for i in idx_params
            if num_harmonic[i] == 0
                #println(num_harmonic[i])
                num_harmonic[i] = 1
            end
        end
        # length(idx_params) > length(num_harmonic) #considera os mesmos harmonicos para todos os parametros variantes, para não quebrar a update_S!
        # num_harmonic = Int64.(ones(length(idx_params)) * num_harmonic[1])
    end

    #println(num_harmonic)

    # if sum(vcat(order...)) == 0
    #     first_idx = 2
    # else 
    #     first_idx = maximum(vcat(order...)) + 1
    # end

    pred_y = zeros(T_fitted + steps_ahead, num_scenarios)
    pred_y[1:T_fitted, :] .= y[first_idx:end]

    Random.seed!(123)
    for t in 1:steps_ahead
        for s in 1:num_scenarios
            for i in idx_params
                update_score!(dict_hyperparams_and_fitted_components, pred_y, d, dist_code, i, T_fitted + t, s)
                if has_random_walk(random_walk, i)
                    update_rw!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_random_walk_slope(random_walk_slope, i)
                    update_rws!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_seasonality(seasonality, i)
                    update_S!(dict_hyperparams_and_fitted_components, num_harmonic, T - T_fitted, i, T_fitted + t, s)
                end
                if has_AR(ar, i)
                    update_AR!(dict_hyperparams_and_fitted_components, order, i, T_fitted + t, s)
                end
                update_params!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s, dist; combination=combination)
            end
            # println("Param = ", dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s])
            # println("Pred_y = ", sample_dist(dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s], dist))
            pred_y[T_fitted + t, s] = sample_dist(dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s], dist)
        end
    end

    return pred_y, dict_hyperparams_and_fitted_components
end

"
Simulates scenarios considering the uncertaintity in the dynamics. Case with exogenous variables.
"
function simulate(gas_model::GASModel, output::Output, dict_hyperparams_and_fitted_components::Dict{String, Any}, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64; combination::String="additive") where {Fl}
    
    @unpack dist, time_varying_params, d, random_walk, random_walk_slope, ar, seasonality, robust, stochastic, combination = gas_model

    idx_params      = get_idxs_time_varying_params(time_varying_params) 
    order           = get_AR_order(ar)
    num_harmonic, _ = get_num_harmonic_and_seasonal_period(seasonality)
    dist_code       = get_dist_code(dist)

    T        = length(y)
    T_fitted = length(output.fit_in_sample)

    first_idx = T - T_fitted + 1

    ### PORQUE RECALCULAR O NUM_HARMONIC?
    if isempty(num_harmonic) #caso não tenha sazonalidade, assume 1 harmonico para nao quebrar a função update_S!
        num_harmonic = Int64.(ones(get_num_params(dist)))
    else
        for i in idx_params
            if num_harmonic[i] == 0
                #println(num_harmonic[i])
                num_harmonic[i] = 1
            end
        end
    end

    # if sum(vcat(order...)) == 0
    #     first_idx = 2
    # else 
    #     first_idx = maximum(vcat(order...)) + 1
    # end

    pred_y = zeros(T_fitted + steps_ahead, num_scenarios)
    pred_y[1:T_fitted, :] .= y[first_idx:end]

    Random.seed!(123)
    for t in 1:steps_ahead
        for s in 1:num_scenarios
            for i in idx_params
                update_score!(dict_hyperparams_and_fitted_components, pred_y, d, dist_code, i, T_fitted + t, s)
                if has_random_walk(random_walk, i)
                    update_rw!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_random_walk_slope(random_walk_slope, i)
                    update_rws!(dict_hyperparams_and_fitted_components, i, T_fitted + t, s)
                end
                if has_seasonality(seasonality, i)
                    update_S!(dict_hyperparams_and_fitted_components, num_harmonic, T - T_fitted, i, T_fitted + t, s)
                end
                if has_AR(ar, i)
                    update_AR!(dict_hyperparams_and_fitted_components, order, i, T_fitted + t, s)
                end
                update_params!(dict_hyperparams_and_fitted_components, X_forecast, t, i, T_fitted + t, s; combination=combination)
            end
            pred_y[T_fitted + t, s] = sample_dist(dict_hyperparams_and_fitted_components["params"][:, T_fitted + t, s], dist)
        end
    end

    return pred_y
end

"
Returns the point forecast as the mean of the scenarios and the specified probabilistic intervals.
"
function get_mean_and_intervals_prediction(pred_y::Matrix{Fl}, steps_ahead::Int64, probabilistic_intervals::Vector{Float64}) where Fl
    
    forec           = zeros(steps_ahead)
    forec_intervals = zeros(steps_ahead, length(probabilistic_intervals) * 2)

    dict_forec = Dict{String, Any}()
    dict_forec["intervals"] = Dict{String, Any}()

    for t in 1:steps_ahead
        forec[t] = mean(pred_y[end - steps_ahead + 1:end, :][t, :])

        for q in eachindex(probabilistic_intervals)
            α = round(1 - probabilistic_intervals[q], digits = 4)
            forec_intervals[t, 2 * q - 1] = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], α/2)
            forec_intervals[t, 2 * q]     = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], 1 - α/2)
        end
    end

    for q in eachindex(probabilistic_intervals)
        α = Int64(100 * probabilistic_intervals[q])
        dict_forec["intervals"]["$α"] = Dict{String, Any}()
        dict_forec["intervals"]["$α"]["upper"] = forec_intervals[:, 2 * q]
        dict_forec["intervals"]["$α"]["lower"] = forec_intervals[:, 2 * q - 1]
    end

    dict_forec["mean"] = forec
    dict_forec["scenarios"] = pred_y[end - steps_ahead + 1:end, :]

    return dict_forec
end

"
Performs the forecast of the GAS model.
"
function predict(gas_model::GASModel, output::Output, y::Vector{Float64}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95], combination::String="additive")
    
    
    dict_hyperparams_and_fitted_components         = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, output, steps_ahead, num_scenarios; combination=combination)
    pred_y, dict_hyperparams_and_fitted_components = simulate(gas_model, output, dict_hyperparams_and_fitted_components, y, steps_ahead, num_scenarios; combination=combination)

    dict_forec = get_mean_and_intervals_prediction(pred_y, steps_ahead, probabilistic_intervals)

    return dict_forec, dict_hyperparams_and_fitted_components
end

"
Performs the forecast of the GAS model for the casa with exogenous variables
"
function predict(gas_model::GASModel, output::Output, y::Vector{Float64}, X_forecast::Matrix{Fl}, steps_ahead::Int64, num_scenarios::Int64; probabilistic_intervals::Vector{Float64} = [0.8, 0.95], combination::String="additive") where {Ml, Fl}
    
    dict_hyperparams_and_fitted_components = get_dict_hyperparams_and_fitted_components_with_forecast(gas_model, output, X_forecast, steps_ahead, num_scenarios; combination=combination)
    pred_y                                 = simulate(gas_model, output, dict_hyperparams_and_fitted_components, y, X_forecast, steps_ahead, num_scenarios; combination=combination)

    dict_forec = get_mean_and_intervals_prediction(pred_y, steps_ahead, probabilistic_intervals)

    return dict_forec
end