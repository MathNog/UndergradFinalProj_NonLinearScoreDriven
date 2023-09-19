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
"
function score_gama(α, λ, y) 
  
    α<0 ? α = 1e-4 : nothing
    λ<0 ? λ = 1e-4 : nothing

    ∇_α =  log(y) - y/λ + log(α) - Ψ1(α) - log(λ) + 1
    ∇_λ = (α/λ)*(y/λ-1)
    println("----------- Score Gamma -------------")
    println(∇_α, ∇_λ)
    return [∇_α; ∇_λ]
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
"
function fisher_information_gama(α, λ) 

    α<0 ? α = 1e-4 : nothing
    λ<0 ? λ = 1e-4 : nothing
    
    I_λ = α/λ^2
    I_α = Ψ2(α) - 1/α

    println("--------------Fisher Gamma --------------")
    println(I_α, I_λ)
    return [I_α 0; 0 I_λ]
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function logpdf_gama(α, λ, y)

    println("--------- LogPDF ---------------")
    println(logpdf_gama([α, λ], y))
    return logpdf_gama([α, λ], y)
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
    param[1] = α
    param[2] = λ 
"
function logpdf_gama(param, y)

    param[1]>=0 ? α = param[1] : α = 1e-4
    param[2]>=0 ? λ = param[2] : λ = 1e-4
    
    return -log(Γ(α)) - α*log(1/α) - α*log(λ) +(α-1)*log(y) - (α/λ)*y
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
    return rand(Distributions.Gamma(param[1], param[2]/param[1]))
end

"
Indicates which parameters of the Normal distribution must be positive.
"
function check_positive_constrainst(dist::GammaDistribution)
    return [true, true]
end


"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"

"ERRO ---- Quem eu devo colocar como parametros iniciais????????"
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::GammaDistribution) where Fl

    println("Inicialização dos parâmetros iniciais")
    T         = length(y)
    dist_code = get_dist_code(dist)

    initial_params = Dict()

    # param[2] = λ = média
    if time_varying_params[2]
        println("λ = y")
        initial_params[2] = y
    else
        println("λ = mean(y)")
        initial_params[2] = mean(y)
    end

    # param[1] = α
    if time_varying_params[1]
        println("α = ??")
        initial_params[1] = (scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        println("α = λ²/var(diff(y))")
        initial_params[1] = mean(y)^2/var(diff(y)) 
    end
    println(length(initial_params))
    println([size(i) for i in values(initial_params)])
    return initial_params
end
 
 