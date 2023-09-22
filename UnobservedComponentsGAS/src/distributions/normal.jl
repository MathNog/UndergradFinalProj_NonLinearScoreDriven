"
Defines a Normal distribution with mean μ and variance σ².
"
mutable struct NormalDistribution <: ScoreDrivenDistribution
    μ::Union{Missing, Float64}
    σ²::Union{Missing, Float64}
end

"
Outer constructor for the Normal distribution.
"
function NormalDistribution()
    return NormalDistribution(missing, missing)
end

"
Evaluate the score of a Normal distribution with mean μ and variance σ², in observation y.
"
function score_normal(μ, σ², y) 
  
    return [(y - μ)/σ²; -(0.5/σ²) * (1 - ((y - μ)^2)/σ²)]
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
"
function fisher_information_normal(μ, σ²)

    return [1/(σ²) 0; 0 1/(2*(σ²^2))]
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function logpdf_normal(μ, σ², y)

    return logpdf_normal([μ, σ²], y)
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
    param[1] = μ
    param[2] = σ² 
"
function logpdf_normal(param, y)

    if param[2] < 0
        param[2] = 1e-4
    end

    return -0.5 * log(2 * π * param[2]) - ((y - param[1])^2)/(2 * (param[2]))
end

"
Evaluate the cdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function cdf_normal(param::Vector{Float64}, y::Fl) where Fl

    return Distributions.cdf(Normal(param[1], sqrt(param[2])), y)
end

"
Returns the code of the Normal distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::NormalDistribution)
    return 1
end

"
Returns the number of parameters of the Normal distribution.
"
function get_num_params(dist::NormalDistribution)
    return 2
end

"
Simulates a value from a given Normal distribution.
    param[1] = μ
    param[2] = σ² 
"
function sample_dist(param::Vector{Float64}, dist::NormalDistribution)
    
    return rand(Normal(param[1], sqrt(param[2])))
end

"
Indicates which parameters of the Normal distribution must be positive.
"
function check_positive_constrainst(dist::NormalDistribution)
    return [false, true]
end


"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::NormalDistribution) where Fl

    T         = length(y)
    dist_code = get_dist_code(dist)

    initial_params = Dict()

    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = mean(y)
    end

    if time_varying_params[2]
        initial_params[2] = (scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = var(diff(y))
    end

    return initial_params
end
 
 