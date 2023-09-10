function slope_matrix(T::Int64)
    slope = Matrix{Float64}(undef, T, T)
    
    for t in 0:T-1
        slope[:, t + 1] = vcat(zeros(t), collect(1:T - t))
    end
    
    return slope
end

function level_matrix(T::Int64)
    level = Matrix{Float64}(undef, T, T)

    for t in 0:T - 1
        level[:, t+1] = vcat(zeros(t), ones(T - t))
    end
    
    return level
end

function outlier_matrix(T::Int64)
    
    return Matrix(1.0 * I, T, T)
end

function seasonal_matrix(T::Int64, s::Int64, n_harmonic::Int64)
    
    seasonality = n_harmonic != s / 2 ?  Matrix{Float64}(undef, T, 2 * n_harmonic) : Matrix{Float64}(undef, T, 2 * n_harmonic - 1)

    idx = repeat(collect(1:s), T ÷ s + 1)[1:T]
    for t in 1:T
        col = 1
        for h = 1:n_harmonic
            if h != s / 2
                seasonality[t, col]     = sin(2 * π * h * idx[t] / s) 
                seasonality[t, col + 1] = cos(2 * π * h * idx[t] / s) 
            else
                seasonality[t, col]     = cos(2 * π * h * idx[t] / s) 
            end
            col += 2
        end
    end
        
    return seasonality
end

function deterministic_seasonality(idx::Vector{Int64}, T::Int64, s::Int64)
    seasonality = zeros(Float64, T, s-1)
    for t in 1:T
        if idx[t] != s
            seasonality[t, idx[t]] = 1.0
        end
    end
    return seasonality
end

function lag_indexes(idx::Int64, s::Int64)
    if idx == 1
        return s, s - 1
    elseif idx == 2
        return 1, s
    else
        return idx - 1, idx - 2
    end
end

function stochastic_seasonality(idx::Vector{Int64}, T::Int64, s::Int64)
    seasonality = zeros(Float64, T, T-1)
    for t in 1:T
        lag1, lag2 = lag_indexes(idx[t] , s)
        lags1 = findall(i -> (i == lag1), idx)
        lags2 = findall(i -> (i == lag2), idx)
        seasonality[t, lags1[lags1 .< t]] .= 1.0
        seasonality[t, lags2[lags2 .< t]] .= -1.0
    end
    return seasonality
end

function seasonal_matrix(T::Int64, s::Int64)

    idx = repeat(collect(1:s), T ÷ s + 1)[1:T]

    return hcat(deterministic_seasonality(idx, T, s), stochastic_seasonality(idx, T, s))

end

function build_components_X(T::Int64, slope::Bool, level::Bool, outlier::Bool, seasonality::Bool; s::Int64 = 12)

    components_X = Matrix{Float64}(undef, T, 0)

    components_X = level ? hcat(components_X, level_matrix(T)) : components_X

    components_X = slope ? hcat(components_X, slope_matrix(T)) : components_X

    components_X = seasonality ? hcat(components_X, seasonal_matrix(T, s)) : components_X

    components_X = outlier ? hcat(components_X, outlier_matrix(T)) : components_X

    components_X += rand(Normal(0,1), size(components_X,1), size(components_X,2))./10000000

    return components_X 
end

function build_components_X(X::Matrix{Tl}, slope::Bool, level::Bool, outlier::Bool, seasonality::Bool; s::Int64 = 12) where {Tl}
    
    T = size(X, 1)

    components_X = build_components_X(T, slope, level, outlier, seasonality; s=s)

    return hcat(components_X, X) 
end