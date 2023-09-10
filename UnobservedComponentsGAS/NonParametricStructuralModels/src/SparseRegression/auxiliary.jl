include("../SubsetSelection/recover_primal.jl")

function get_information_criteria(T::Int64, K::Int64, ϵ::Vector{Float64}; criteria::String = "bic")
    if criteria == "bic"
        return T*log(var(ϵ)) + K*log(T)
    elseif criteria == "aic"
        return 2*K + T*log(var(ϵ))
    elseif criteria == "aicc"
        return 2*K + T*log(var(ϵ))  + ((2*K^2 +2*K)/(T - K - 1))
    else
        throw("unavailable method")
    end
end

function get_opt_γ(y::Vector{Float64}, X::Matrix{Float64})
    T,p = size(X)
    ratio = p > T ? 0.01*(p/T) : 0.0001
    cv_results = GLMNet.glmnetcv(X, y, alpha=0, nfolds=5, lambda_min_ratio=ratio, standardize=true)
    return cv_results.lambda[argmin(cv_results.meanloss)]
end 

function get_Sparse_Regressor(y::Vector{Float64}, X::Matrix{Float64}, K::Int64, γ::Float64, fixed_idx::Vector{Int64})
    return subsetSelection(OLS(), Constraint(K), y, X; γ = γ, fixed_idx=fixed_idx)
end

function get_starter_bic(y::Vector{Float64}, X::Matrix{Float64}, γ::Float64, T::Int64, p::Int64, criteria::String, fixed_idx::Vector{Int64})
    start_β = recover_primal(OLS(), y, X[:,fixed_idx], γ)
    β = zeros(p); β[fixed_idx] = start_β
    ϵ = y - X*β
    return get_information_criteria(T, length(fixed_idx), ϵ; criteria = criteria), β
end

function get_outlier_model(y::Vector{Float64}, X::Matrix{Float64}, p::Int64, T::Int64, γ::Float64, outlier_dict::Dict, indexes::Vector{Int64})

    outlier_indexes  = Int64[]
    for i in indexes
        if i in keys(outlier_dict)
            push!(outlier_indexes, outlier_dict[i])
        end
    end
    ny = y[findall(i -> !(i in outlier_indexes), collect(1:T))]
    nX = X[findall(i -> !(i in outlier_indexes), collect(1:T)), :]
    non_outlier_idx = indexes[findall(i -> !(i in keys(outlier_dict)), indexes)]
    final_β = recover_primal(OLS(), ny, nX[:,non_outlier_idx], γ)
   
    β = zeros(p); β[non_outlier_idx] = final_β
    ϵ                                = y - X*β
    ϵ[outlier_indexes]              .= 0

    for i in indexes
        if i in keys(outlier_dict)
            β[i] = (y - X*β)[outlier_dict[i]]
        end
    end

    return β, ϵ
end

function get_relaxed_results(y::Vector{Float64}, X::Matrix{Float64}, p::Int64, T::Int64, K::Int64, γ::Float64, criteria::String; fixed_idx::Vector{Int64}=Int64[], outlier_dict::Dict=Dict())
    Sparse_Regressor         = get_Sparse_Regressor(y, X, K, γ, fixed_idx)
    indexes                  = Sparse_Regressor.indices

    β, ϵ = get_outlier_model(y, X, p, T, γ, outlier_dict, indexes)
    return get_information_criteria(T, K, ϵ; criteria = criteria), β, indexes
end

function get_exact_results(y::Vector{Float64}, X::Matrix{Float64}, p::Int64, T::Int64, K::Int64, γ::Float64, criteria::String; fixed_idx::Vector{Int64}=Int64[], outlier_dict::Dict=Dict())
    relaxed_indexes = get_Sparse_Regressor(y, X, K, γ, fixed_idx).indices
    exact_output    = SubsetSelectionCIO.oa_formulation(OLS(), y, X, K, γ; fixed_idx=fixed_idx, indices0=relaxed_indexes)
    indexes         = exact_output[1]
    
    β = zeros(p); β[indexes] = exact_output[2]
    ϵ                        = y - X*β
    return get_information_criteria(T, K, ϵ; criteria = criteria), β, indexes
end

function get_results(y::Vector{Float64}, X::Matrix{Float64}, p::Int64, T::Int64, K::Int64, γ::Float64, criteria::String, exact_problem::Bool; fixed_idx::Vector{Int64}=Int64[], outlier_dict::Dict=Dict())
    if exact_problem
        return get_exact_results(y, X, p, T, K, γ, criteria; fixed_idx = fixed_idx, outlier_dict = outlier_dict)
    else
        return get_relaxed_results(y, X, p, T, K, γ, criteria; fixed_idx = fixed_idx, outlier_dict = outlier_dict)
    end
end