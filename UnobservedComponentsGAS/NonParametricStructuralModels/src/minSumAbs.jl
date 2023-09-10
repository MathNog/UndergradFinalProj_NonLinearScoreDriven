#= function minimize_sum_abs(y::Vector{Fl}, T::Int64) where {Fl}

    X = hcat(ones(T), collect(1:T))
    model = Model(Tulip.Optimizer)
    set_silent(model)

    # Define variables
    p = size(X, 2)
    @variable(model, β[1:p])
    @variable(model, θ⁺[1:T] >= 0)
    @variable(model, θ⁻[1:T] >= 0)

    # Define constraints
    @constraint(model, [i = 1:T], y[i] == X[i, :]'*β  + θ⁺[i] - θ⁻[i])

    # Define objective
    @objective(model, Min, sum(θ⁺ + θ⁻))

    # Solve model
    optimize!(model)

    return value.(θ⁺) - value.(θ⁻)
end
 =#
function trimmmed_transformation(y::Vector{Float64})

    T = length(y)
    shift_factor = median(y[1:Int(floor(0.1*T))])

    return y .- shift_factor, shift_factor
end

function fix_transformation!(coefs::Vector{Float64}, components::Dict, 
                                shift_factor::Float64)

    coefs[1] += shift_factor
    components["level"]["values"] .+= shift_factor
end