function get_information(T::Int64, K::Int64, ϵ::Vector{Float64}; method::String = "bic")
    if method == "bic"
        return T*log(var(ϵ)) + K*log(T)
    elseif method == "aic"
        return 2*K + T*log(var(ϵ))
    elseif method == "aicc"
        return 2*K + T*log(var(ϵ))  + ((2*K^2 +2*K)/(T - K - 1))
    else
        throw("unavailable method")
    end
end

function get_K(coefs_matrix::CompressedPredictorMatrix)
    return count(i->i != 0, coefs_matrix; dims = 1)'
end

function get_path_information_criteria(model::GLMNetPath, X::Matrix, y::Vector, method::String)
    path_size = length(model.lambda)
    T         = length(y)
    K         = get_K(model.betas)

    method_vec = Vector{Float64}(undef, path_size)
    for i in 1:path_size
        fit = X*model.betas[:, i] .+ model.a0[i]
        ϵ   = y - fit
        
        method_vec[i] = get_information(T, K[i], ϵ; method = method)
    end

    best_model_idx = argmin(method_vec)
    coefs = vcat(model.a0[best_model_idx], model.betas[:, best_model_idx])
    fit   = hcat(ones(T), X)*coefs
    ϵ   = y - fit
    return coefs, ϵ, fit
end

function fit_glmnet(X::Matrix, y::Vector, alpha::Float64; method::String = "cv", penalty_factor::Vector{Float64}=ones(size(X,2)))
    if method == "cv"
        model = glmnetcv(X, y, alpha = alpha, penalty_factor = penalty_factor)
        β     = vcat(model.path.a0[argmin(model.meanloss)], GLMNet.coef(model))
        fit   = hcat(ones(length(y)), X)*β
        ϵ     = y - fit
        return β, ϵ, fit
    else
        model = glmnet(X, y, alpha = alpha, penalty_factor = penalty_factor)
        return get_path_information_criteria(model, X, y, method)
    end
end