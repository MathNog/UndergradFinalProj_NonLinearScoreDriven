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

            end
        end
    end
end

df_mapes = hcat(coluna_comb, coluna_d, coluna_distribuicao, coluna_serie, coluna_mape_treino, coluna_mape_teste)
df_mapes = DataFrame(df_mapes,:auto)
rename!(df_mapes, colunas)
CSV.write(current_path*"\\Saidas\\Relatorio\\df_mapes.csv", df_mapes)

" Compila Gráficos para o Relatório"

dir2 = current_path*"\\Saidas\\CombNaoLinear\\Multiplicative1\\d_1\\Gamma\\"

img = load(dir2*"airline_forecast_Gamma.png")

for combination in combinacoes
    for d in ds
        for distribution in distributions
            for serie in series
                dir = current_path*"\\Saidas\\CombNaoLinear\\$combination\\$(dict_d[d])\\$distribution\\"

                if serie*"_fit_in_sample_"*distribution*""
                end

            end
        end
    end
end
