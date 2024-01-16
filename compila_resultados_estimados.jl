using CSV, Plots, DataFrames, HypothesisTests, Distributions


" Obtendo dados dos modelos estimados "

"C:\\Users\\matheuscn.ELE\\Desktop\\TCC Matheus\\TCC\\Saidas\\CombNaoLinear\\SazoDeterministica\\additive\\Gamma"

current_path = pwd()

path_saidas = current_path * "\\Saidas\\CombNaoLinear\\SazoDeterministica\\"

combinations = ["additive\\", "multiplicative1\\", "multiplicative2\\", "multiplicative3"]

distributions = ["LogNormal\\", "Gamma\\"]



df_fitted_values = CSV.read(path_saidas*combinations[1]*distributions[1]*"carga_fitted_values.csv", DataFrame)


residuals = rand(100)

test_H(residuals;type="julia")
test_H(residuals;type="distrib")


