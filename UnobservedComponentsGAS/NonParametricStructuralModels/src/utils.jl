function get_components(coefs::Vector{Float64}, X::Matrix{Float64}, slope::Bool, level::Bool, outlier::Bool, 
                        seasonality::Bool, explanatory::Bool; s::Int64 = 12)

    T, p= size(X)
    components = get_components_cols(slope, level, outlier, seasonality, explanatory, s, T, p)

    n_coefs = !level ? coefs[2:end] : deepcopy(coefs)

    components["level"]["values"] = level ? X[:, components["level"]["idx"]]*n_coefs[components["level"]["idx"]] : ones(T).*coefs[1]

    slope ? components["slope"]["values"] = X[:, components["slope"]["idx"]]*n_coefs[components["slope"]["idx"]] : nothing
  
    slope && level ? components["trend"]["values"] = components["level"]["values"] + components["slope"]["values"] : nothing

    seasonality ? components["seasonality"]["values"] = X[:, components["seasonality"]["idx"]]*n_coefs[components["seasonality"]["idx"]] : nothing
  
    outlier ? components["outlier"]["values"] = X[:, components["outlier"]["idx"]]*n_coefs[components["outlier"]["idx"]] : nothing
  
    explanatory ? components["explanatory"]["values"] = X[:, components["explanatory"]["idx"]]*n_coefs[components["explanatory"]["idx"]] : nothing

    return components
end

function recover_components!(components::Dict, y::Vector{Fl}) where {Fl}

    for k in keys(components)
        components[k]["values"] = recover_data(y, components[k]["values"])
    end

end

function get_fit_in_sample(y::Vector{Float64}, res::Vector{Float64})

    return y .- res
end

function find_repeated_cols(X::Matrix{Tl}) where {Tl}

    T,p= size(X)

    repeated_cols = Int64[]
    for i in 1:p
        if X[:, i] == vcat(zeros(T - 1), ones(1))
            push!(repeated_cols, i)
        end
    end

    if length(repeated_cols) == 1
        return Int64[]
    else
        return repeated_cols
    end
end

function find_last_repeated_col(X::Matrix{Tl}) where {Tl}

    T,p= size(X)

    repeated_cols = Int64[]
    for i in 1:p
        if X[:, i] == vcat(zeros(T - 1), ones(1))
            push!(repeated_cols, i)
        end
    end

    return repeated_cols[end]
end

function find_col(X::Matrix{Tl}) where {Tl}
    T, p= size(X)

    col = nothing
    for i in 1:p
        if X[:, i] == vcat(zeros(T - 1), ones(1))
            col = i
        end
    end
    return col
end

function get_components_cols(slope::Bool, level::Bool, outlier::Bool, seasonality::Bool, 
                             explanatory::Bool, s::Int64, T::Int64, p::Int64)

    components = Dict{String, Dict{String, Any}}()
    components["slope"] = Dict{String, Any}()
    components["level"] = Dict{String, Any}()
    components["seasonality"] = Dict{String, Any}()
    components["outlier"] = Dict{String, Any}()
    components["trend"] = Dict()

    actual_idx = 0

    level ? components["level"]["idx"] = (actual_idx+1:actual_idx+T) : actual_idx -= T
    actual_idx += T
    
    slope ? components["slope"]["idx"] = (actual_idx+1:actual_idx+T) : actual_idx -= T
    actual_idx += T

    seasonality ? components["seasonality"]["idx"] = (actual_idx+1:actual_idx+T + s - 2) : actual_idx -= T + s - 2
    actual_idx += T + s - 2

    outlier ? components["outlier"]["idx"] = (actual_idx+1:actual_idx+T) : actual_idx -= T
    actual_idx += T

    explanatory ? components["explanatory"] = Dict() : nothing
    explanatory ? components["explanatory"]["idx"] = (actual_idx+1:p) : nothing

    return components
end

function recover_data(y::Vector{Fl}, normalized_y::Vector{Float64}) where{Fl}

    return (normalized_y .* (maximum(y) - minimum(y))) .+ minimum(y)
end

function start_component_factor!(penalty_factor::Vector{Float64}, slope::Bool, level::Bool, 
                                outlier::Bool, seasonality::Bool, explanatory::Bool, 
                                s::Int64, T::Int64, p::Int64)
    components = get_components_cols(slope, level, outlier, seasonality, explanatory, s, T, p)
    adjust_intercept = level ? 1 : 0
    slope ? penalty_factor[components["slope"]["idx"][1]-adjust_intercept] = 0 : nothing
    seasonality ? penalty_factor[components["seasonality"]["idx"][1:s-1].-adjust_intercept] .= 0 : nothing
end

function get_variances(coefs::Vector{Float64}, components::Dict, residuals::Vector{Float64}, slope::Bool, level::Bool, seasonality::Bool, s::Int64)
    variances = Dict()
    variances["ε"] = var(residuals)
    slope && level ? variances["ξ"] = var(coefs[components["level"]["idx"][2:end]]) : nothing
    slope ? variances["ζ"] = var(coefs[components["slope"]["idx"][2:end]]) : nothing
    seasonality ? variances["ω"] = var(coefs[components["seasonality"]["idx"][s:end]]) : nothing
    return variances
end

function get_residuals(coefs, X, y)
    return y - X*coefs
end