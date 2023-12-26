using CSV, DataFrames, FileIO

current_path = pwd()

combinacoes = ["multiplicative1", "multiplicative2"]

dict_d = Dict(0.0 => "d_0", 0.5 => "d_05", 1.0 => "d_1")
ds = keys(dict_d)

distributions = ["LogNormal", "Gamma"]

series = ["ena", "airline", "carga"]

colunas = ["Combinacao", "d", "Distribuicao", "Serie", "MAPE Treino", "MAPE Teste"]
coluna_comb = []
coluna_d = []
coluna_serie = []
coluna_distribuicao = []
coluna_mape_treino = []
coluna_mape_teste = []

coluna_teste_jb = []
coluna_pvalor_jb = []
coluna_rejeicao_jb = []

coluna_teste_arch = []
coluna_pvalor_arch = []
coluna_rejeicao_arch = []

coluna_teste_lb = []
coluna_pvalor_lb = []
coluna_rejeicao_lb = []

for combination in combinacoes
    for d in ds
        for distribution in distributions
            for serie in series
                dir = current_path*"\\Saidas\\CombNaoLinear\\$combination\\$(dict_d[d])\\$distribution\\"

                if "$(serie)_mapes.csv" in readdir(dir)
                    df_mape = CSV.read(dir*"$(serie)_mapes.csv", DataFrame)

                    mape_treino = round(df_mape[1,"MAPE Treino"], digits=2)
                    mape_teste = round(df_mape[1,"MAPE Teste"], digits=2)

                    push!(coluna_comb, combination)
                    push!(coluna_d, d)
                    push!(coluna_serie, serie)
                    push!(coluna_distribuicao, distribution)
                    push!(coluna_mape_treino, mape_treino)
                    push!(coluna_mape_teste, mape_teste)
                end

                # if "$(serie)_quantile_residuals_diagnostics_05.csv" in readdir(dir)
                #     df_diag = CSV.read(dir*"$(serie)_quantile_residuals_diagnostics_05.csv", DataFrame)

                    
                #     push!(coluna_comb, combination)
                #     push!(coluna_d, d)
                #     push!(coluna_serie, serie)
                #     push!(coluna_distribuicao, distribution)

                #     push!(coluna_teste_jb, df_diag[1, "Teste"])
                #     push!(coluna_pvalor_jb, round.(df_diag[1, "pvalor"], digits = 4))
                #     push!(coluna_rejeicao_jb, df_diag[1, "Rejeicao"])

                #     push!(coluna_teste_arch, df_diag[2, "Teste"])
                #     push!(coluna_pvalor_arch, round.(df_diag[2, "pvalor"], digits = 4))
                #     push!(coluna_rejeicao_arch, df_diag[2, "Rejeicao"])

                #     push!(coluna_teste_lb, df_diag[3, "Teste"])
                #     push!(coluna_pvalor_lb, round.(df_diag[3, "pvalor"], digits = 4))
                #     push!(coluna_rejeicao_lb, df_diag[3, "Rejeicao"])
                # end

            end
        end
    end
end

df_mapes = hcat(coluna_comb, coluna_d, coluna_distribuicao, coluna_serie, coluna_mape_treino, coluna_mape_teste)
df_mapes = DataFrame(df_mapes,:auto)
rename!(df_mapes, colunas)
CSV.write(current_path*"\\Saidas\\Relatorio\\df_mapes.csv", df_mapes)

df_jb   = DataFrame(hcat(coluna_comb, coluna_d, coluna_distribuicao, coluna_serie, coluna_teste_jb, coluna_pvalor_jb, coluna_rejeicao_jb), :auto)
df_arch = DataFrame(hcat(coluna_comb, coluna_d, coluna_distribuicao, coluna_serie, coluna_teste_arch, coluna_pvalor_arch, coluna_rejeicao_arch), :auto)
df_lb   = DataFrame(hcat(coluna_comb, coluna_d, coluna_distribuicao, coluna_serie, coluna_teste_lb, coluna_pvalor_lb, coluna_rejeicao_lb), :auto)
rename!(df_jb, ["Combinacao", "d", "Distribuicao", "Serie", "Teste", "pvalor", "Rejeicao"])
rename!(df_arch, ["Combinacao", "d", "Distribuicao", "Serie", "Teste", "pvalor", "Rejeicao"])
rename!(df_lb, ["Combinacao", "d", "Distribuicao", "Serie", "Teste", "pvalor", "Rejeicao"])
CSV.write(current_path*"\\Saidas\\Relatorio\\df_jarquebera.csv", df_jb)
CSV.write(current_path*"\\Saidas\\Relatorio\\df_archlm.csv", df_arch)
CSV.write(current_path*"\\Saidas\\Relatorio\\df_ljungbox.csv", df_lb)