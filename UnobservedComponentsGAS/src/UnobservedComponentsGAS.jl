module UnobservedComponentsGAS

    using JuMP
    using Ipopt
    using LinearAlgebra
    using Parameters
    using Statistics
    using Distributions
    using Random 
    using SpecialFunctions
    using Optim
     
    include("../NonParametricStructuralModels/src/NonParametricStructuralModels.jl")

    include("structures.jl")
    include("distributions/common.jl")
    include("distributions/normal.jl")
    include("distributions/gama.jl")
    include("distributions/t_location_scale.jl")
    include("initialization.jl")
    include("fit.jl")
    include("utils.jl")
    include("components_dynamics.jl")
    include("optimization.jl")
    include("forecast.jl")
    include("update_model.jl")

    const DICT_CODE = Dict(1 => "Normal",
                           2 => "tLocationScale",
                           3 => "Gamma" )

    const DICT_SCORE = Dict("Normal"         => score_normal,
                            "tLocationScale" => score_tlocationscale,
                            "Gamma"          => score_gama)

    const DICT_FISHER_INFORMATION = Dict("Normal"         => fisher_information_normal,
                                         "tLocationScale" => fisher_information_tlocationscale,
                                         "Gamma"          => fisher_information_gama)

    const DICT_LOGPDF = Dict("Normal"         => logpdf_normal,
                             "tLocationScale" => logpdf_tlocationscale,
                             "Gamma"          => logpdf_gama)

    const DICT_CDF = Dict("Normal"         => cdf_normal,
                          "tLocationScale" => cdf_tlocationscale,
                          "Gamma"          => cdf_gama)
end 