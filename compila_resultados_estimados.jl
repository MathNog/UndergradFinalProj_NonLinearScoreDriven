using CSV, Plots, DataFrames, HypothesisTests, Distributions, StatsBase


function correct_scale(series, K, residuals)
    SQR = sum(residuals.^2)
    N = length(residuals)
    σ² = SQR/(N-K)
    return exp.(series)*exp(0.5*σ²)
end

" Definindo nomes"

current_path = pwd()
path_dados = current_path * "\\Saidas\\CombNaoLinear\\SazoDeterministica\\"
path_series =  current_path * "\\Dados\\Tratados\\"
path_saida = current_path * "\\Saidas\\Relatorio\\"

combinations  = ["additive\\", "multiplicative2\\", "multiplicative3\\"]
distributions = ["LogNormal\\", "Gamma\\"]
series        = ["ena", "carga", "uk_visits"]

nomes_series        = Dict("ena"=>"ENA", "carga"=> "carga", "uk_visits"=>"viagens")
nomes_combinacoes   = Dict("additive"=>"ad", "multiplicative2"=>"mult1", "multiplicative3"=> "mult2")
nomes_distribuicoes = Dict("LogNormal"=>"lognormal", "Gamma"=>"gama")
num_parameters = Dict("ena"=>17, "carga"=>17, "uk_visits"=>17)

"---------------- Criando dicionario com séries -----------"

dict_series = Dict()

ena = CSV.read(path_series*"ena_limpo.csv", DataFrame)
carga = CSV.read(path_series*"carga_limpo.csv", DataFrame)
viagens = CSV.read(path_series*"uk_visits.csv", DataFrame)[:,2:end]

len_teste = 12

dict_series["ena"] = Dict()
dict_series["ena"]["serie_treino"] = ena[:,"ENA"][1:end-12]
dict_series["ena"]["serie_teste"]  = ena[:,"ENA"][end-11:end]
dict_series["ena"]["datas_treino"] = ena[:,"Data"][1:end-12]
dict_series["ena"]["datas_teste"]  = ena[:,"Data"][end-11:end]

dict_series["carga"] = Dict()
dict_series["carga"]["serie_treino"] = carga[:,"Carga"][1:end-12]
dict_series["carga"]["serie_teste"]  = carga[:,"Carga"][end-11:end]
dict_series["carga"]["datas_treino"] = carga[:,"Data"][1:end-12]
dict_series["carga"]["datas_teste"]  = carga[:,"Data"][end-11:end]

dict_series["uk_visits"] = Dict()
dict_series["uk_visits"]["serie_treino"] = viagens[:,"Valor"][1:end-12]
dict_series["uk_visits"]["serie_teste"]  = viagens[:,"Valor"][end-11:end]
dict_series["uk_visits"]["datas_treino"] = viagens[:,"Data"][1:end-12]
dict_series["uk_visits"]["datas_teste"]  = viagens[:,"Data"][end-11:end]



"---------------- Criando dicionario com valores estimados, residuos e previsao -----------"

dict_fit_in_sample = Dict()
dict_residuos      = Dict()
dict_std_residuos  = Dict()
dict_q_residuos    = Dict()
dict_forecast      = Dict()
dict_datas_teste   = Dict()
dict_datas_treino  = Dict()


for combination in combinations
    combination_name = nomes_combinacoes[combination[1:end-1]]
    dict_fit_in_sample[combination_name] = Dict()
    dict_residuos[combination_name]      = Dict()
    dict_std_residuos[combination_name]  = Dict()
    dict_q_residuos[combination_name]    = Dict()
    dict_forecast[combination_name]      = Dict()
    dict_datas_teste[combination_name]   = Dict()
    dict_datas_treino[combination_name]  = Dict()

    for distribution in distributions
        distribution_name = nomes_distribuicoes[distribution[1:end-1]]
        dict_fit_in_sample[combination_name][distribution_name] = Dict()
        dict_residuos[combination_name][distribution_name] = Dict()
        dict_std_residuos[combination_name][distribution_name] = Dict()
        dict_q_residuos[combination_name][distribution_name] = Dict()
        dict_forecast[combination_name][distribution_name] = Dict()
        dict_datas_teste[combination_name][distribution_name] = Dict()
        dict_datas_treino[combination_name][distribution_name] = Dict()
        
        for serie in series
            
            df_fitted_values = CSV.read(path_dados*combination*distribution*"$(serie)_fitted_values.csv", DataFrame)
            dict_fit_in_sample[combination_name][distribution_name][serie] = df_fitted_values[:,"fit_in_sample"]
            dict_residuos[combination_name][distribution_name][serie]      = df_fitted_values[:,"residuals"]
            dict_std_residuos[combination_name][distribution_name][serie]  = df_fitted_values[:,"std_residuals"]
            dict_q_residuos[combination_name][distribution_name][serie]    = df_fitted_values[:,"q_residuals"]
            dict_datas_treino[combination_name][distribution_name][serie]  = df_fitted_values[:,"dates"]
            
            df_forecast = CSV.read(path_dados*combination*distribution*"$(serie)_forecast_values.csv", DataFrame)
            dict_forecast[combination_name][distribution_name][serie]      = df_forecast[:,"mean"]
            dict_datas_teste[combination_name][distribution_name][serie]   = df_forecast[:,"dates"]
        end
    end
end



"Corrigindo a escala do fit in sample"

for (combination, combination_name) in nomes_combinacoes
    for serie in series
        K = num_parameters[serie]
        combination == "multiplicative3" ? K = num_parameters[serie] : K = num_parameters[serie]-1
        dict_fit_in_sample[combination_name]["lognormal"][serie] = correct_scale(dict_fit_in_sample[combination_name]["lognormal"][serie],
                                                                                K, dict_residuos[combination_name]["lognormal"][serie])
        dict_forecast[combination_name]["lognormal"][serie] = correct_scale(dict_forecast[combination_name]["lognormal"][serie],
                                                                                K, dict_residuos[combination_name]["lognormal"][serie])
    end
end

"---------------- Criando dicionario com parametros estimados -----------"
dict_params_carga   = Dict()
dict_params_ena     = Dict()
dict_params_viagens = Dict()

for combination in combinations
    combination_name = nomes_combinacoes[combination[1:end-1]]
    dict_params_carga[combination_name]   = Dict()
    dict_params_ena[combination_name]     = Dict()
    dict_params_viagens[combination_name] = Dict()

    
    for distribution in distributions
        distribution_name = nomes_distribuicoes[distribution[1:end-1]]
        distribution == "Gamma\\" ? param = "param_2_" : param = "param_1_"

        dict_params_carga[combination_name][distribution_name]   = Dict()
        dict_params_ena[combination_name][distribution_name]     = Dict()
        dict_params_viagens[combination_name][distribution_name] = Dict()

        df_params_carga   = CSV.read(path_dados*combination*distribution*"carga_params.csv", DataFrame)
        df_params_ena     = CSV.read(path_dados*combination*distribution*"ena_params.csv", DataFrame)
        df_params_viagens = CSV.read(path_dados*combination*distribution*"uk_visits_params.csv", DataFrame)

        dict_params_carga[combination_name][distribution_name]["κ_level"] = df_params_carga[:,param*"level_κ"]
        dict_params_carga[combination_name][distribution_name]["κ_slope"] = df_params_carga[:,param*"slope_κ"]
        dict_params_carga[combination_name][distribution_name]["ϕ_slope"] = df_params_carga[:,param*"slope_ϕb"]
        dict_params_carga[combination_name][distribution_name]["b_mult"] = df_params_carga[:,param*"b_mult"]

        dict_params_ena[combination_name][distribution_name]["κ_ar"] = df_params_ena[:,param*"ar_κ"]
        dict_params_ena[combination_name][distribution_name]["ϕ_ar"] = df_params_ena[:,param*"ar_ϕ"]
        dict_params_ena[combination_name][distribution_name]["b_mult"] = df_params_ena[:,param*"b_mult"]
        
        dict_params_viagens[combination_name][distribution_name]["κ_level"] = df_params_viagens[:,param*"level_κ"]
        dict_params_viagens[combination_name][distribution_name]["κ_slope"] = df_params_viagens[:,param*"slope_κ"]
        dict_params_viagens[combination_name][distribution_name]["ϕ_slope"] = df_params_viagens[:,param*"slope_ϕb"]
        dict_params_viagens[combination_name][distribution_name]["b_mult"] = df_params_viagens[:,param*"b_mult"] 
    end
end


" ------------ Criando gráficos de fit in sample, residuos, fac e previsao ---------------"


for serie in series
    println("$serie")
    for (combination, combination_name) in nomes_combinacoes
        println("   $(combination_name)")
        for (distribution, distribution_name) in nomes_distribuicoes
            println("       $(distribution_name)")

            plot(title="Fit in sample - $(nomes_series[serie]) - $(distribution_name) - $(combination_name)")
            plot!(dict_series[serie]["datas_treino"], dict_series[serie]["serie_treino"], label="Série")
            plot!(dict_datas_treino[combination_name][distribution_name][serie], dict_fit_in_sample[combination_name][distribution_name][serie], label="Valores Estimados")
            savefig(path_saida*"\\ResultadosIndividuais\\$(combination_name)\\$(distribution)\\$(serie)_fit_in_sample_$(distribution).png")

            plot(title="Previsão 12 passos à frente - $(nomes_series[serie]) - $(distribution_name) - $(combination_name)", titlefontsize=12)
            plot!(dict_series[serie]["datas_teste"], dict_series[serie]["serie_teste"], label="Série")
            plot!(dict_datas_teste[combination_name][distribution_name][serie], dict_forecast[combination_name][distribution_name][serie], label="Previsão", color="red")
            savefig(path_saida*"\\ResultadosIndividuais\\$(combination_name)\\$(distribution)\\$(serie)_forecast_$(distribution).png")

            res = dict_q_residuos[combination_name][distribution_name][serie][2:end]
            acf_values = autocor(res)[1:15]
            lag_values = collect(0:14)
            conf_interval = 1.96 / sqrt(length(res)-1)  # 95% confidence interval

            plot(title="FAC dos Residuos Quantílicos - $(nomes_series[serie]) - $(distribution_name) - $(combination_name)", titlefontsize=12)
            plot!(lag_values,acf_values,seriestype=:stem, label="",xticks=(lag_values,lag_values))
            hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
            savefig(path_saida*"\\ResultadosIndividuais\\$(combination_name)\\$(distribution)\\$(serie)_quantile_residuals_acf_$(distribution).png")

        end
    end
end

" ---------------- Criando arquivos de parametros estimados -------------------"

colunas = ["κ_slope", "κ_level", "ϕ_slope", "b_mult"]
col_comb = []
col_distrib = []
df_params_carga = DataFrame([col => [] for col in colunas])
for (combination, combination_name) in nomes_combinacoes
    df_ln = DataFrame(dict_params_carga[combination_name]["lognormal"])
    df_g = DataFrame(dict_params_carga[combination_name]["gama"])
    col_comb = vcat(col_comb, [combination_name, combination_name])
    col_distrib = vcat(col_distrib, ["lognormal", "gama"])
    df_params_carga = vcat(df_params_carga, vcat(df_ln, df_g))
end

df_params_carga = round.(df_params_carga, digits=5)

df_params_carga = hcat(DataFrame("combinacao"=>col_comb, "distribuicao"=>col_distrib), df_params_carga)
CSV.write(path_saida*"Resultados\\params_carga.csv", df_params_carga)

colunas = ["κ_slope", "κ_level", "ϕ_slope", "b_mult"]
col_comb = []
col_distrib = []
df_params_viagens = DataFrame([col => [] for col in colunas])
for (combination, combination_name) in nomes_combinacoes
    df_ln = DataFrame(dict_params_viagens[combination_name]["lognormal"])
    df_g = DataFrame(dict_params_viagens[combination_name]["gama"])
    col_comb = vcat(col_comb, [combination_name, combination_name])
    col_distrib = vcat(col_distrib, ["lognormal", "gama"])
    df_params_viagens = vcat(df_params_viagens, vcat(df_ln, df_g))
end
df_params_viagens = round.(df_params_viagens, digits=5)
df_params_viagens = hcat(DataFrame("combinacao"=>col_comb, "distribuicao"=>col_distrib), df_params_viagens)
CSV.write(path_saida*"Resultados\\params_viagens.csv", df_params_viagens)



colunas = ["b_mult", "κ_ar", "ϕ_ar"]
col_comb = []
col_distrib = []
df_params_ena = DataFrame([col => [] for col in colunas])
for (combination, combination_name) in nomes_combinacoes
    df_ln = DataFrame(dict_params_ena[combination_name]["lognormal"])
    df_g = DataFrame(dict_params_ena[combination_name]["gama"])
    col_comb = vcat(col_comb, [combination_name, combination_name])
    col_distrib = vcat(col_distrib, ["lognormal", "gama"])
    df_params_ena = vcat(df_params_ena, vcat(df_ln, df_g))
end

#Entender porque está vindo uma linha a mais
df_params_ena = round.(df_params_ena, digits=5)

df_params_ena = hcat(DataFrame("combinacao"=>col_comb, "distribuicao"=>col_distrib), df_params_ena)
CSV.write(path_saida*"Resultados\\params_viagens.csv", df_params_carga)


" ---------------- Criando arquivos de testes de hipoteses -------------------"


" ---------------- Criando arquivos de mapes -------------------"


# "---------------- Criando gráficos de dispersão -----------"

# for serie in series
#     for distribution in distributions
#         fit_add = dict_fit_in_sample["additive\\"]["$(distribution)"][serie]
#         fit_m1 = dict_fit_in_sample["multiplicative1\\"]["$(distribution)"][serie]
#         fit_m2 = dict_fit_in_sample["multiplicative2\\"]["$(distribution)"][serie]
#         fit_m3 = dict_fit_in_sample["multiplicative3\\"]["$(distribution)"][serie]

#         p0 = plot(fit_add, fit_add, seriestype=:scatter, title="Fit in sample add X add", xlabel = "add", ylabel = "add", label="")
#         p1 = plot(fit_add, fit_m1, seriestype=:scatter, title="Fit in sample add X mult1", xlabel = "add", ylabel = "mult1", label="")
#         p2 = plot(fit_add, fit_m2, seriestype=:scatter, title="Fit in sample add X mult2", xlabel = "add", ylabel = "mult2", label="")
#         p3 = plot(fit_add, fit_m3, seriestype=:scatter, title="Fit in sample add X mult3", xlabel = "add", ylabel = "mult3", label="")

#         plot(p1,p2,p3, layout=grid(3,1), size=(800,800), plot_title = "$(distribution) - $(serie)")
#         savefig("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\TCC\\Saidas\\Slides\\dispersao_fit_$(serie)_$(distribution[1:end-1]).png")

#     end
# end    

