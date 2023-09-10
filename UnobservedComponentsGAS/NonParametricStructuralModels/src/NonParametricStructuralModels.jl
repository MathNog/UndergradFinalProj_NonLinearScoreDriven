module NonParametricStructuralModels

using Lasso, LinearAlgebra, Statistics, LIBLINEAR, Compat, Random, GLMNet, Distributions

include("structures.jl")
include("create_matrices.jl")
include("minSumAbs.jl")
include("information_criteria.jl")
include("utils.jl")
include("fit_model.jl")
include("forecast.jl")


include("SubsetSelection/types.jl")
include("SubsetSelection/stepsize.jl")
include("SubsetSelection/recover_primal.jl")
include("SubsetSelection/SubsetSelection.jl")

include("SparseRegression/auxiliary.jl")
include("SparseRegression/SparseRegression.jl")

end # module NonParametricStructuralModels
