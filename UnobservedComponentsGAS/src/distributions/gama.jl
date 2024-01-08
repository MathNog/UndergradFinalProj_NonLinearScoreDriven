"
Defines a Gamma distribution with parameters α λ.
From a shape (α) and ratio (β) parametrization, we obtain our parametrization making λ = α/β
"
mutable struct GammaDistribution <: ScoreDrivenDistribution
    α::Union{Missing, Float64}
    λ::Union{Missing, Float64}
end

"
Outer constructor for the Normal distribution.
"
function GammaDistribution()
    return GammaDistribution(missing, missing)
end

"
Gamma Function Γ(x)
"
function Γ(x)
    return SpecialFunctions.gamma(x)
end
    

"Auxiliar function Ψ1(α)"
function Ψ1(α)
    return SpecialFunctions.digamma(α)
end


"Auxiliar function Ψ2(α)"
function Ψ2(α)
    return SpecialFunctions.trigamma(α)
end


"
Evaluate the score of a Normal distribution with mean μ and variance σ², in observation y.
Colocar link aqui
"
function score_gama(α, λ, y) 
  
    α <= 0 ? α = 1e-2 : nothing
    λ <= 0 ? λ = 1e-4 : nothing

    # ∇_α =  log(y) - y/λ + log(α) - Ψ1(α) - log(λ) + 1
    # ∇_λ = (α/λ)*((y/λ)-1)

    ∇_α_exp = α * (log(y) - y/λ + log(α) - Ψ1(α) - log(λ) + 1)
    ∇_λ_exp = α * (y/λ - 1)
    # println("----------- Score Gamma -------------")
    # println(∇_α, ∇_λ)
    return [∇_α_exp; ∇_λ_exp]
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
Colocar link aqui
"
function fisher_information_gama(α, λ) 

    α <= 0 ? α = 1e-2 : nothing
    λ <= 0 ? λ = 1e-4 : nothing
    
    # I_λ = α/(λ^2)
    # I_α = Ψ2(α) - 1/α

    I_α_exp = α^2 * (Ψ2(α) - 1/α)
    I_λ_exp = α

    # println("--------------Fisher Gamma --------------")
    # println(I_α, I_λ)
    return [I_α_exp 0; 0 I_λ_exp]
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function logpdf_gama(α, λ, y)

    # println("--------- LogPDF ---------------")
    # println(logpdf_gama([λ, α], y))
    return logpdf_gama([α, λ], y)
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
    param[1] = α
    param[2] = λ
"
function logpdf_gama(param, y)

    param[1] > 0 ? α = param[1] : α = 1e-2
    param[2] > 0 ? λ = param[2] : λ = 1e-2
    # println("y = ", y)
    # println("Expressao = ",-log(Γ(α)) - α*log(λ) + α*log(α) + (α-1)*log(y) - (α/λ)*y)
    # println("Distributions = ", Distributions.logpdf(Distributions.Gamma(α, λ/α), y))
    # println("α = $(α)")
    # println("λ = $(λ)")
    # return -log(Γ(α)) - α*log(λ) + α*log(α) + (α-1)*log(y) - (α/λ)*y
    return Distributions.logpdf(Distributions.Gamma(α, λ/α), y)
end

"
Evaluate the cdf of a Gamma distribution with α,λ, in observation y.
"
function cdf_gama(param::Vector{Float64}, y::Fl) where Fl

    param[1] > 0 ? α = param[1] : α = 1e-2
    param[2] > 0 ? λ = param[2] : λ = 1e-2

    return Distributions.cdf(Distributions.Gamma(α, λ/α), y)
end

"
Returns the code of the Normal distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::GammaDistribution)
    return 3
end

"
Returns the number of parameters of the Normal distribution.
"
function get_num_params(dist::GammaDistribution)
    return 2
end

"
Simulates a value from a given Normal distribution.
    param[1] = α
    param[2] = λ  
"
function sample_dist(param::Vector{Float64}, dist::GammaDistribution)
    
    "A Gamma do pacote Distributions é parametrizada com shape α e scale θ"
    "Como θ = 1/β e β = α/λ, segue-se que θ = λ/α"
    # println("α = $(param[1]) || λ = $(param[2])")
    param[1] > 0 ? α = param[1] : α = 1e-2
    param[2] > 0 ? λ = param[2] : λ = 1e-2
    println(param[2])
    return rand(Distributions.Gamma(α, λ/α))
end

"
Indicates which parameters of the Normal distribution must be positive.
"
function check_positive_constrainst(dist::GammaDistribution)
    return [true, true]
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

"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::GammaDistribution, seasonality::Union{Dict{Int64, Int64}, Dict{Int64, Bool}}) where Fl

    println("Inicialização dos parâmetros iniciais")
    T         = length(y)
    dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

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
 
 
function get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::GammaDistribution) where Fl

    num_periods = ceil(Int, length(y) / seasonal_period)
    seasonal_variances = zeros(Fl, length(y))
    
    for i in 1:seasonal_period
        month_data = y[i:seasonal_period:end]
        num_observations = length(month_data)
        if num_observations > 0
            g = Distributions.fit(Gamma, month_data)
            α = g.α
            θ = g.θ
            variance = α*(θ^2) 
            for j in 1:num_observations
                seasonal_variances[i + (j - 1) * seasonal_period] = variance
            end
        end
    end
    return seasonal_variances
end
 