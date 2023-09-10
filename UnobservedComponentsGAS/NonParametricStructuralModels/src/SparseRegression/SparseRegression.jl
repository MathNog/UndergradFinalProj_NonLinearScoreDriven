using LinearAlgebra, GLMNet, Statistics, Distributions

function SparseRegressionfit(y::Vector{Float64}, X::Matrix{Float64}; γ::Float64=NaN, criteria::String = "aic", 
            fixed_idx::Vector{Int64}=Int64[], outlier_dict::Dict=Dict())
   
    T, p    = size(X)
    K       = 1 + length(fixed_idx)
    keep    = true

    @info("Executing grid search for the optimal γ given K = p")
    γ = isnan(γ) ? 1/sqrt(T) : γ
    @info("optimal γ: "*string(γ))
    
    best_info, β = get_starter_bic(y, X, γ, T, p, criteria, fixed_idx)

    β_hist = [β]; indexes_hist = [fixed_idx]; info_hist = [best_info]
    #@info("K: "*string(K-1))
    #@info("current bic: "*string(best_info))
    while keep
        @info("K: "*string(K-length(fixed_idx)))

        current_info, β, indexes = get_results(y, X, p, T, K, γ, criteria, false; fixed_idx = fixed_idx, outlier_dict = outlier_dict)
        
        if !isempty(info_hist)
            if current_info > best_info && info_hist[end] > best_info
                keep = false  
            end
        end
        current_info < best_info ? best_info = current_info : nothing
        push!(β_hist, β); push!(indexes_hist, indexes); push!(info_hist, current_info)
        @info("current bic: "*string(current_info))
        K += 1
    end
    best_info_index = argmin(info_hist) 

    return β_hist[best_info_index], indexes_hist[best_info_index]
end
