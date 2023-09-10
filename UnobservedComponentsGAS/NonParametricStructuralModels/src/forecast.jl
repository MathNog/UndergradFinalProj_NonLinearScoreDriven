function forecast(output::Output, steps_ahead::Int64; extrapolate_last_obs::Int64 = 3)

    coefs          = output.coefs
    complete_X     = output.X
    components     = output.components

    complete_X_forecast = get_complete_X_forecast(steps_ahead, complete_X, components, extrapolate_last_obs)

    return complete_X_forecast * coefs, get_components_forecast(complete_X_forecast, output, extrapolate_last_obs)
end

function forecast(output::Output, X_forecast::Matrix{Fl}, steps_ahead::Int64; extrapolate_last_obs::Int64 = 3) where {Fl}

    coefs          = output.coefs
    complete_X     = output.X
    components     = output.components

    complete_X_forecast = get_complete_X_forecast(steps_ahead, complete_X, components, X_forecast, extrapolate_last_obs)

    return complete_X_forecast * coefs, get_components_forecast(complete_X_forecast, output, extrapolate_last_obs)
end

function get_complete_X_forecast(steps_ahead::Int64, complete_X::Matrix{Fl}, components::Dict, extrapolate_last_obs::Int64) where {Fl}

    T, p                = size(complete_X)
    complete_X_forecast = Matrix{Float64}(undef, steps_ahead, p)

    T_all = T + steps_ahead

    haskey(components["level"], "idx") ? complete_X_forecast[:, components["level"]["idx"]] = level_matrix(T_all)[T+1:end, 1:T] : nothing

    haskey(components["slope"], "idx") ? complete_X_forecast[:, components["slope"]["idx"]] = slope_matrix(T_all)[T+1:end, 1:T] : nothing

    haskey(components["seasonality"], "idx") ? s = length(components["seasonality"]["idx"]) - T + 2 : nothing
    haskey(components["seasonality"], "idx") ? complete_X_forecast[:, components["seasonality"]["idx"]] = seasonal_matrix(T_all, s)[T+1:end, 1:T+s-2] : nothing
    
    haskey(components["outlier"], "idx") ? complete_X_forecast[:, components["outlier"]["idx"]] = zeros(steps_ahead, length(components["outlier"]["idx"])) : nothing

    if haskey(components["level"], "idx")
        return complete_X_forecast
    else
        return hcat(ones(steps_ahead), complete_X_forecast)
    end
end

function get_complete_X_forecast(steps_ahead::Int64, complete_X::Matrix{Fl}, components::Dict, X_forecast::Matrix{Fl}, extrapolate_last_obs::Int64) where {Fl}

    T, p                = size(complete_X)
    complete_X_forecast = Matrix{Float64}(undef, steps_ahead, p)

    T_all = T + steps_ahead

    haskey(components["level"], "idx") ? complete_X_forecast[:, components["level"]["idx"]] = level_matrix(T_all)[T+1:end, 1:T] : nothing

    haskey(components["slope"], "idx") ? complete_X_forecast[:, components["slope"]["idx"]] = slope_matrix(T_all)[T+1:end, 1:T] : nothing

    haskey(components["seasonality"], "idx") ? s = length(components["seasonality"]["idx"]) - T + 2 : nothing
    haskey(components["seasonality"], "idx") ? complete_X_forecast[:, components["seasonality"]["idx"]] = seasonal_matrix(T_all, s)[T+1:end, 1:T+s-2] : nothing
    
    haskey(components["outlier"], "idx") ? complete_X_forecast[:, components["outlier"]["idx"]] = zeros(steps_ahead, length(components["outlier"]["idx"])) : nothing

    @assert haskey(components, "explanatory")
    @assert size(X_forecast, 1) == steps_ahead
    complete_X_forecast[:, components["explanatory"]["idx"]] = X_forecast

    if haskey(components["level"], "idx")
        return complete_X_forecast
    else
        return hcat(ones(steps_ahead), complete_X_forecast)
    end
end

function get_components_forecast(complete_X_forecast::Matrix{Fl}, output::Output, extrapolate_last_obs::Int64) where {Fl}

    coefs          = output.coefs
    components     = output.components

    components_forecast = Dict{String, Vector{Float64}}()

    steps_ahead = size(complete_X_forecast, 1)

    n_coefs = !haskey(components["level"], "idx") ? coefs[2:end] : deepcopy(coefs)
    adjust_indexes = !haskey(components["level"], "idx") ? 1 : 0

    components_forecast["level"] = haskey(components["level"], "idx") ? complete_X_forecast[:, components["level"]["idx"]]*n_coefs[components["level"]["idx"]] : ones(steps_ahead).*coefs[1]

    haskey(components["slope"], "idx") ? components_forecast["slope"] = complete_X_forecast[:, components["slope"]["idx"].+adjust_indexes]*n_coefs[components["slope"]["idx"]] : nothing
  
    haskey(components["seasonality"], "idx") ? components_forecast["seasonality"] = complete_X_forecast[:, components["seasonality"]["idx"].+adjust_indexes]*n_coefs[components["seasonality"]["idx"]] : nothing
  
    haskey(components["outlier"], "idx") ? components_forecast["outlier"] = complete_X_forecast[:, components["outlier"]["idx"].+adjust_indexes]*n_coefs[components["outlier"]["idx"]] : nothing
  
    haskey(components, "explanatory") ? components_forecast["explanatory"] = complete_X_forecast[:, components["explanatory"]["idx"].+adjust_indexes]*n_coefs[components["explanatory"]["idx"]] : nothing

    haskey(components["slope"], "idx") && haskey(components["level"], "idx") ? components_forecast["trend"] = components_forecast["level"] + components_forecast["slope"] : nothing

    return components_forecast
end
