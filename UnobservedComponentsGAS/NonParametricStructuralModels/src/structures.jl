mutable struct Output{Tl, Fl} 
    components::Dict{String, Dict{String, Any}} 
    selected_variables::Vector{Int64}
    X::Matrix{Tl}
    fit::Vector{Fl}
    residuals::Vector{Fl}
    coefs::Vector{Float64}
    y::Vector{Fl}
    variances::Dict
end