function fit_model(y::Vector{Fl}; slope::Bool = true, level::Bool = true, seasonality::Bool = true, 
            outlier::Bool = true, s::Int64 = 12, α::Float64 = 0.5, model::String = "AdaLasso", method::String = "cv") where {Fl}

    T = length(y)
    @assert T >= 2*s

    @assert (model == "AdaLasso" || model == "Lasso" || model == "Heuristic Integer" || model == "Exact Integer")
    
    X    = build_components_X(T, slope, level, outlier, seasonality; s = s)

    p = size(X, 2)

    if model == "Lasso" || model == "AdaLasso"
        penalty_factor = ones(p)

        Estimate_X = level ? X[:, 2:end] : X
        penalty_factor = level ? penalty_factor[2:end] : penalty_factor
        start_component_factor!(penalty_factor, slope, level, outlier, seasonality, false, s, T, p)
        
        #lasso_model    = Lasso.fit(LassoModel, Estimate_X, y; α = α, penalty_factor = penalty_factor, criterion=:obj)
        #glm_model = glmnetcv(Estimate_X, y; alpha = α, penalty_factor = penalty_factor)
        coefs, res, _ = fit_glmnet(Estimate_X, y, α; method = method, penalty_factor = penalty_factor)
   
        if model == "AdaLasso"
            #penalty_factor = 1 ./ (0.0001 .+ abs.(GLMNet.coef(glm_model)[2:end])); 
            penalty_factor = 1 ./ (0.0001 .+ abs.(coefs[2:end])); 
            start_component_factor!(penalty_factor, slope, level, outlier, seasonality, false, s, T, p)
            #lasso_model = Lasso.fit(LassoModel, Estimate_X, y; α = α, penalty_factor = penalty_factor, criterion=:obj) 
            #glm_model = glmnetcv(Estimate_X, y; alpha = α, penalty_factor = penalty_factor)
            coefs, res, _ = fit_glmnet(Estimate_X, y, α; method = method, penalty_factor = penalty_factor)
        end
        #coefs = vcat(glm_model.path.a0[argmin(glm_model.meanloss)], GLMNet.coef(glm_model))
        #res   = get_residuals(coefs, hcat(ones(T), Estimate_X), y)
        #res   = Lasso.residuals(glm_model)
    elseif model == "Heuristic Integer"
        
        norm_y, shift_factor = trimmmed_transformation(y)

        components = get_components_cols(slope, level, outlier, seasonality, false, s, T, p)

        intercept_factor = !level ? 1 : 0
        Estimate_X = level ? X : hcat(ones(T), X)
        outlier_idx = Vector(components["outlier"]["idx"]) .+ intercept_factor

        fixed_idx=setdiff(collect(1:p+intercept_factor), outlier_idx)
        outlier_dict = Dict()
        for i in outlier_idx
            outlier_dict[i] = findfirst(i->i ==1, Estimate_X[:,i])
        end

        coefs, selected_variables = SparseRegressionfit(norm_y, Estimate_X; γ=1/sqrt(T),criteria="bic", fixed_idx=fixed_idx,outlier_dict=outlier_dict)
        res = Estimate_X*coefs - norm_y
    else
        @error("Model not implemented")
    end

    components = get_components(coefs, X, slope, level, outlier, seasonality, false; s = s)
    
    selected_variables = findall(i -> i != 0.0, coefs)
    fitted_values      = (get_fit_in_sample(y, res))

    variances = get_variances(coefs, components, res, slope, level, seasonality, s)

    model == "Heuristic Integer" ? fix_transformation!(coefs, components, shift_factor) : nothing
    
    return Output(components, selected_variables, X, fitted_values, res, coefs, y, variances)
end


function fit_model(y::Vector{Fl}, Exogenous_X::Matrix{Tl}; slope::Bool = true, level::Bool = true, seasonality::Bool = true, outlier::Bool = true, 
                            s::Int64 = 12, α::Float64 = 0.5, model::String = "AdaLasso", method::String = "cv") where {Fl, Tl}
    
    T = length(y)

    @assert T >= 2*s

    X    = build_components_X(Exogenous_X, slope, level, outlier, seasonality; s = s)

    p = size(X, 2)
  
    if model == "Lasso" || model == "AdaLasso"
        penalty_factor = ones(p)

        Estimate_X = level ? X[:, 2:end] : X
        penalty_factor = level ? penalty_factor[2:end] : penalty_factor
        start_component_factor!(penalty_factor, slope, level, outlier, seasonality, false, s, T, p)
        
        #lasso_model    = Lasso.fit(LassoModel, Estimate_X, y; α = α, penalty_factor = penalty_factor, criterion=:obj)
        coefs, res, _ = fit_glmnet(Estimate_X, y, α; method = method, penalty_factor = penalty_factor)
   
        if model == "AdaLasso"
            penalty_factor = 1 ./ (0.0001 .+ abs.(coefs[2:end])); 
            start_component_factor!(penalty_factor, slope, level, outlier, seasonality, false, s, T, p)
            #lasso_model = Lasso.fit(LassoModel, Estimate_X, y; α = α, penalty_factor = penalty_factor, criterion=:obj) 
            #glm_model = glmnetcv(Estimate_X, y; alpha = α, penalty_factor = penalty_factor)
            coefs, res, _ = fit_glmnet(Estimate_X, y, α; method = method, penalty_factor = penalty_factor)
        end
        #coefs = vcat(glm_model.path.a0[argmin(glm_model.meanloss)], GLMNet.coef(glm_model))
        #res   = get_residuals(coefs, hcat(ones(T), Estimate_X), y)
        #res   = Lasso.residuals(glm_model)
    elseif model == "Heuristic Integer"

        norm_y, shift_factor = trimmmed_transformation(y)
        
        components = get_components_cols(slope, level, outlier, seasonality, true, s, T, p)
        intercept_factor = !level ? 1 : 0
        Estimate_X = level ? X : hcat(ones(T), X)
        outlier_idx = Vector(components["outlier"]["idx"]) .+ intercept_factor
        explanatory_idx = Vector(components["explanatory"]["idx"]) .+ intercept_factor

        fixed_idx=setdiff(collect(1:p+intercept_factor), vcat(outlier_idx, explanatory_idx))
        outlier_dict = Dict()
        for i in outlier_idx
            outlier_dict[i] = findfirst(i->i ==1, Estimate_X[:,i])
        end

        coefs, selected_variables = SparseRegressionfit(norm_y, Estimate_X; γ=1/sqrt(T),criteria="bic", fixed_idx=fixed_idx,outlier_dict=outlier_dict)
        res = Estimate_X*coefs - norm_y
    else
        @error("Model not implemented")
    end
    
    components = get_components(coefs, X, slope, level, outlier, seasonality, true; s = s)

    selected_variables = findall(i -> i != 0.0, coefs)
    fitted_values      = get_fit_in_sample(y, res)

    variances = get_variances(coefs, components, res, slope, level, seasonality, s)

    model == "Heuristic Integer" ? fix_transformation!(coefs, components, shift_factor) : nothing
    
    return Output(components, selected_variables, X, fitted_values, res, coefs, y, variances)
end
