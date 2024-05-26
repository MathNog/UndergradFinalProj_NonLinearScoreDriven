using CSV, Plots, DataFrames, HypothesisTests, Distributions, StatsBase, MLJBase


function correct_scale(series, K, residuals)
    SQR = sum(residuals.^2)
    N = length(residuals)
    σ² = SQR/(N-K)
    return exp.(series)*exp(0.5*σ²)
end

function MASE(y, y_hat)
    numerator   = MLJBase.mae(y_hat, y)
    prev_naive = diff(y)
    denominator = mean(abs.(prev_naive))
    return numerator / denominator
end


" Definindo nomes"

current_path = pwd()
path_dados = current_path * "\\Saidas\\CombNaoLinear\\SazoDeterministica\\"
path_series =  current_path * "\\Dados\\Tratados\\"
path_saida = current_path * "\\Saidas\\SBPO\\"

combinations  = ["additive\\", "multiplicative3\\"]
distributions = ["LogNormal\\", "Gamma\\"]
series        = ["uk_visits"]

nomes_series        = Dict("uk_visits"=>"uk travel")
nomes_combinacoes   = Dict("additive"=>"additive", "multiplicative3"=> "non linear")
nomes_distribuicoes = Dict("LogNormal"=>"lognormal", "Gamma"=>"gamma")
num_parameters = Dict("uk_visits"=>17)

"---------------- Criando dicionario com séries -----------"

dict_series = Dict()
viagens = CSV.read(path_series*"uk_visits.csv", DataFrame)[:,2:end]

len_teste = 12

dict_series["uk_visits"] = Dict()
dict_series["uk_visits"]["serie_treino"] = viagens[:,"Valor"][1:end-12]
dict_series["uk_visits"]["serie_teste"]  = viagens[:,"Valor"][end-11:end]
dict_series["uk_visits"]["datas_treino"] = viagens[:,"Data"][1:end-12]
dict_series["uk_visits"]["datas_teste"]  = viagens[:,"Data"][end-11:end]


function get_mean_and_intervals_prediction(pred_y::Matrix{Fl}, steps_ahead::Int64, probabilistic_intervals::Vector{Float64}) where Fl
    
    forec           = zeros(steps_ahead)
    forec_intervals = zeros(steps_ahead, length(probabilistic_intervals) * 2)

    dict_forec = Dict{String, Any}()
    dict_forec["intervals"] = Dict{String, Any}()

    for t in 1:steps_ahead
        forec[t] = mean(pred_y[end - steps_ahead + 1:end, :][t, :])

        for q in eachindex(probabilistic_intervals)
            α = round(1 - probabilistic_intervals[q], digits = 4)
            forec_intervals[t, 2 * q - 1] = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], α/2)
            forec_intervals[t, 2 * q]     = quantile(pred_y[end - steps_ahead + 1:end, :][t, :], 1 - α/2)
        end
    end

    for q in eachindex(probabilistic_intervals)
        α = Int64(100 * probabilistic_intervals[q])
        dict_forec["intervals"]["$α"] = Dict{String, Any}()
        dict_forec["intervals"]["$α"]["upper"] = forec_intervals[:, 2 * q]
        dict_forec["intervals"]["$α"]["lower"] = forec_intervals[:, 2 * q - 1]
    end

    dict_forec["mean"] = forec
    dict_forec["scenarios"] = pred_y[end - steps_ahead + 1:end, :]

    return dict_forec
end





"---------------- Criando dicionario com valores estimados, residuos e previsao -----------"

dict_fit_in_sample = Dict()
dict_residuos      = Dict()
dict_std_residuos  = Dict()
dict_q_residuos    = Dict()
dict_forecast      = Dict()
dict_scenarios      = Dict()
dict_datas_teste   = Dict()
dict_datas_treino  = Dict()

dict_testes_hipotese = Dict()
dict_mapes           = Dict()

for combination in combinations
    combination_name = nomes_combinacoes[combination[1:end-1]]
    dict_fit_in_sample[combination_name] = Dict()
    dict_residuos[combination_name]      = Dict()
    dict_std_residuos[combination_name]  = Dict()
    dict_q_residuos[combination_name]    = Dict()
    dict_forecast[combination_name]      = Dict()
    dict_scenarios[combination_name]      = Dict()
    dict_datas_teste[combination_name]   = Dict()
    dict_datas_treino[combination_name]  = Dict()

    dict_testes_hipotese[combination_name]   = Dict()
    dict_mapes[combination_name]  = Dict()

    for distribution in distributions
        distribution_name = nomes_distribuicoes[distribution[1:end-1]]
        dict_fit_in_sample[combination_name][distribution_name] = Dict()
        dict_residuos[combination_name][distribution_name] = Dict()
        dict_std_residuos[combination_name][distribution_name] = Dict()
        dict_q_residuos[combination_name][distribution_name] = Dict()
        dict_forecast[combination_name][distribution_name] = Dict()
        dict_scenarios[combination_name][distribution_name] = Dict()
        dict_datas_teste[combination_name][distribution_name] = Dict()
        dict_datas_treino[combination_name][distribution_name] = Dict()

        dict_testes_hipotese[combination_name][distribution_name]   = Dict()
        dict_mapes[combination_name][distribution_name]  = Dict()
        
        serie  = "uk_visits"
            
        df_fitted_values = CSV.read(path_dados*combination*distribution*"$(serie)_fitted_values.csv", DataFrame)
        dict_fit_in_sample[combination_name][distribution_name][serie] = df_fitted_values[:,"fit_in_sample"]
        dict_residuos[combination_name][distribution_name][serie]      = df_fitted_values[:,"residuals"]
        dict_std_residuos[combination_name][distribution_name][serie]  = df_fitted_values[:,"std_residuals"]
        dict_q_residuos[combination_name][distribution_name][serie]    = df_fitted_values[:,"q_residuals"]
        dict_datas_treino[combination_name][distribution_name][serie]  = df_fitted_values[:,"dates"]
        
        df_forecast = CSV.read(path_dados*combination*distribution*"$(serie)_forecast_values.csv", DataFrame)
        dict_forecast[combination_name][distribution_name][serie]      = df_forecast[:,"mean"]
        idx_cols_scenarios = findall(x -> occursin(r"^s\d+$", string(x)), names(df_forecast))
        scenarios = matrix(df_forecast[:,idx_cols_scenarios])
        dict_scenarios[combination_name][distribution_name][serie] = get_mean_and_intervals_prediction(scenarios, 12, [0.80, 0.95])
        dict_datas_teste[combination_name][distribution_name][serie]   = df_forecast[:,"dates"]

        df_testes = CSV.read(path_dados*combination*distribution*"$(serie)_residuals_diagnostics_05.csv", DataFrame)
        dict_testes_hipotese[combination_name][distribution_name][serie] = df_testes

        df_mapes = CSV.read(path_dados*combination*distribution*"$(serie)_mapes.csv", DataFrame)
        dict_mapes[combination_name][distribution_name][serie] = df_mapes
    end
end


dict_scenarios["aditiva"]["gama"]["uk_visits"]

"Corrigindo a escala do fit in sample"

for (combination, combination_name) in nomes_combinacoes
    serie = "uk_visits"
    K = num_parameters[serie]
    combination == "multiplicative3" ? K = num_parameters[serie] : K = num_parameters[serie]-1
    dict_fit_in_sample[combination_name]["lognormal"][serie] = correct_scale(dict_fit_in_sample[combination_name]["lognormal"][serie],
                                                                            K, dict_residuos[combination_name]["lognormal"][serie])
    dict_forecast[combination_name]["lognormal"][serie] = correct_scale(dict_forecast[combination_name]["lognormal"][serie],
                                                                            K, dict_residuos[combination_name]["lognormal"][serie])
    for i in 1:500
        dict_scenarios[combination_name]["lognormal"][serie]["scenarios"][:,i] =correct_scale(dict_scenarios[combination_name]["lognormal"][serie]["scenarios"][:,i],
                                                                                K, dict_residuos[combination_name]["lognormal"][serie])
    end
    dict_scenarios[combination_name]["lognormal"][serie] = get_mean_and_intervals_prediction(dict_scenarios[combination_name]["lognormal"][serie]["scenarios"], 12, [0.80, 0.95])
end

"---------------- Criando dicionario com parametros estimados -----------"

dict_params_viagens = Dict()

for combination in combinations
    combination_name = nomes_combinacoes[combination[1:end-1]]
    dict_params_viagens[combination_name] = Dict()

    for distribution in distributions
        distribution_name = nomes_distribuicoes[distribution[1:end-1]]
        distribution == "Gamma\\" ? param = "param_2_" : param = "param_1_"
       
        dict_params_viagens[combination_name][distribution_name] = Dict()

        df_params_viagens = CSV.read(path_dados*combination*distribution*"uk_visits_params.csv", DataFrame)
    
        dict_params_viagens[combination_name][distribution_name]["κ_level"] = df_params_viagens[:,param*"level_κ"]
        dict_params_viagens[combination_name][distribution_name]["κ_slope"] = df_params_viagens[:,param*"slope_κ"]
        dict_params_viagens[combination_name][distribution_name]["ϕ_slope"] = df_params_viagens[:,param*"slope_ϕb"]
        dict_params_viagens[combination_name][distribution_name]["b_mult"] = df_params_viagens[:,param*"b_mult"] 
    end
end

theme(:ggplot2)

serie = "uk_visits"

plot(title="Time series - uk travel")
plot!(dict_series[serie]["datas_treino"], dict_series[serie]["serie_treino"], color=:black, lw=2, label = "")
plot!(dict_series[serie]["datas_teste"], dict_series[serie]["serie_teste"], color=:black, lw=2  , label = "")
savefig(path_saida*"$(serie).png")


for (combination, combination_name) in nomes_combinacoes
    println("   $(combination_name)")
    for (distribution, distribution_name) in nomes_distribuicoes
        println("       $(distribution_name)")

        plot(title="Fit in sample - $(distribution_name) - $(combination_name)")
        plot!(dict_series[serie]["datas_treino"], dict_series[serie]["serie_treino"], label="time series", color=:black, lw=2)
        plot!(dict_datas_treino[combination_name][distribution_name][serie], dict_fit_in_sample[combination_name][distribution_name][serie], label="fitted values", color=:blue, lw=1, s=:solid)
        savefig(path_saida*"$(serie)_fit_in_sample_$(distribution)_$(combination).png")

        forec = dict_scenarios[combination_name][distribution_name][serie]
        plot(title="Point Forecast - $(distribution_name) - $(combination_name)", titlefontsize=12, legend = :topleft)
        plot!(dict_datas_teste[combination_name][distribution_name][serie], forec["intervals"]["95"]["lower"], fillrange = forec["intervals"]["95"]["upper"], fillalpha = 0.15, color = :grey, label = "95% Confidence band")
        plot!(dict_datas_teste[combination_name][distribution_name][serie], forec["intervals"]["80"]["lower"], fillrange = forec["intervals"]["80"]["upper"], fillalpha = 0.15, color = :darkgrey, label = "80% Confidence band")
        plot!(dict_datas_teste[combination_name][distribution_name][serie], forec["intervals"]["95"]["upper"], label = "", color = :grey)
        plot!(dict_datas_teste[combination_name][distribution_name][serie], forec["intervals"]["80"]["upper"], label = "", color = :darkgrey)
        plot!(dict_series[serie]["datas_teste"], dict_series[serie]["serie_teste"], label="time series", color=:black, lw=2)
        plot!(dict_datas_teste[combination_name][distribution_name][serie], forec["mean"], label="forecasted values", color=:red, lw=1, s=:solid)
        savefig(path_saida*"$(serie)_forecast_$(distribution)_$(combination).png")

        plot(title="Quantile residuals - $(distribution_name) - $(combination_name)")
        plot!(dict_datas_treino[combination_name][distribution_name][serie][3:end], dict_q_residuos[combination_name][distribution_name][serie][3:end],label="", color = :black)
        savefig(path_saida*"$(serie)_quantile_residuals_$(distribution)_$(combination).png")

        res = dict_q_residuos[combination_name][distribution_name][serie][2:end]
        acf_values = autocor(res)[1:15]
        lag_values = collect(0:14)
        conf_interval = 1.96 / sqrt(length(res)-1)  # 95% confidence interval

        plot(title="Quantile residuals ACF - $(distribution_name) - $(combination_name)", titlefontsize=12)
        plot!(lag_values,acf_values,seriestype=:stem, label="",xticks=(lag_values,lag_values), color=:black)
        hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
        savefig(path_saida*"$(serie)_quantile_residuals_acf_$(distribution)_$(combination).png")

    end
end


colunas = ["κ_slope", "κ_level", "ϕ_slope", "b_mult"]
col_comb = []
col_distrib = []
df_params_viagens = DataFrame([col => [] for col in colunas])
for (combination, combination_name) in nomes_combinacoes
    df_ln = DataFrame(dict_params_viagens[combination_name]["lognormal"])
    df_g = DataFrame(dict_params_viagens[combination_name]["gamma"])
    col_comb = vcat(col_comb, [combination_name, combination_name])
    col_distrib = vcat(col_distrib, ["lognormal", "gamma"])
    df_params_viagens = vcat(df_params_viagens, vcat(df_ln, df_g))
end
df_params_viagens = round.(df_params_viagens, digits=5)
df_params_viagens = hcat(DataFrame("combinacao"=>col_comb, "distribuicao"=>col_distrib), df_params_viagens)
rename!(df_params_viagens, replace.(names(df_params_viagens),"κ"=>"kappa"))
rename!(df_params_viagens, replace.(names(df_params_viagens),"ϕ"=>"phi"))
CSV.write(path_saida*"params_uk_visits.csv", df_params_viagens)


df_testes = ones(0,3)

col_comb    = []
col_distrib = []
col_serie   = []

for (combination, combination_name) in nomes_combinacoes
    for (distribution, distribution_name) in nomes_distribuicoes
        serie = "uk_visits"
        col_comb = vcat(col_comb, [combination_name, combination_name, combination_name, combination_name])
        col_distrib = vcat(col_distrib, [distribution_name, distribution_name, distribution_name, distribution_name])
        col_serie = vcat(col_serie, [serie, serie, serie, serie])
        df_testes = vcat(df_testes, Matrix(dict_testes_hipotese[combination_name][distribution_name][serie][:, ["Teste", "pvalor", "Rejeicao"]]))
    end
end
df_testes[:,2] = round.(df_testes[:,2], digits=5)
df_testes = hcat(col_comb, col_distrib, col_serie, df_testes)
df_testes = DataFrame(df_testes, :auto)
rename!(df_testes, ["combinacao", "distribuicao", "serie", "teste", "pvalor", "rejeicao"])
CSV.write(path_saida*"testes_hipoteses_05.csv", df_testes)


df_mapes = ones(0,2)
df_mases = ones(0,2)

col_comb    = []
col_distrib = []
col_serie   = []

for (combination, combination_name) in nomes_combinacoes
    for (distribution, distribution_name) in nomes_distribuicoes
        for serie in series
            col_serie = vcat(col_serie, serie)
            col_comb = vcat(col_comb, combination_name)
            col_distrib = vcat(col_distrib, distribution_name)
            df_mapes = vcat(df_mapes, Matrix(dict_mapes[combination_name][distribution_name][serie][:, ["MAPE Treino", "MAPE Teste"]]))
        end
    end
end
df_mapes = round.(df_mapes, digits=5)
df_mapes = hcat(col_comb, col_distrib, col_serie, df_mapes)
df_mapes = DataFrame(df_mapes, :auto)
rename!(df_mapes, ["combinacao", "distribuicao", "serie", "MAPE Treino", "MAPE Teste"])
CSV.write(path_saida*"mapes.csv", df_mapes)

" ---------------- Criando arquivos de mases -------------------"

df_mases = ones(0,2)

col_comb    = []
col_distrib = []
col_serie   = []

for (combination, combination_name) in nomes_combinacoes
    for (distribution, distribution_name) in nomes_distribuicoes
        for serie in series
            col_serie = vcat(col_serie, serie)
            col_comb = vcat(col_comb, combination_name)
            col_distrib = vcat(col_distrib, distribution_name)
            
            mase_treino = MASE(dict_series[serie]["serie_treino"],dict_fit_in_sample[combination_name][distribution_name][serie])
            mase_teste = MASE(dict_series[serie]["serie_teste"],dict_forecast[combination_name][distribution_name][serie])
            df_mases = vcat(df_mases, hcat(mase_treino, mase_teste))
        end
    end
end

df_mases = round.(df_mases, digits=5)
df_mases = hcat(col_comb, col_distrib, col_serie, df_mases)
df_mases = DataFrame(df_mases, :auto)
rename!(df_mases, ["combinacao", "distribuicao", "serie", "MASE Treino", "MASE Teste"])
CSV.write(path_saida*"mases.csv", df_mases)


df_mases
df_mapes

# Filter the data for the 'uk_visits' series
df_filtered = df_mases[df_mases.serie .== "uk_visits", :]

# Extract the MAPE values for Lognormal and Gamma distributions for Treino and Teste
treino_lognormal = df_filtered[df_filtered.distribuicao .== "lognormal", "MASE Treino"]
treino_gamma = df_filtered[df_filtered.distribuicao .== "gamma", "MASE Treino"]
teste_lognormal = df_filtered[df_filtered.distribuicao .== "lognormal", "MASE Teste"]
teste_gamma = df_filtered[df_filtered.distribuicao .== "gamma", "MASE Teste"]

# Create the new DataFrame with the specified columns and rows
new_df = DataFrame(
    Treino_Lognormal = treino_lognormal,
    Treino_Gamma = treino_gamma,
    Teste_Lognormal = teste_lognormal,
    Teste_Gamma = teste_gamma
)


# Filter the DataFrame for each distribution and combination
lognormal_additive = df_params_viagens[(df_params_viagens.distribuicao .== "lognormal") .& (df_params_viagens.combinacao .== "additive"),    ["kappa_level", "kappa_slope", "phi_slope", "b_mult"]]
lognormal_nonlinear = df_params_viagens[(df_params_viagens.distribuicao .== "lognormal") .& (df_params_viagens.combinacao .== "non linear"), ["kappa_level", "kappa_slope", "phi_slope", "b_mult"]]
gamma_additive = df_params_viagens[(df_params_viagens.distribuicao .== "gamma") .& (df_params_viagens.combinacao .== "additive"),            ["kappa_level", "kappa_slope", "phi_slope", "b_mult"]]
gamma_nonlinear = df_params_viagens[(df_params_viagens.distribuicao .== "gamma") .& (df_params_viagens.combinacao .== "non linear"),         ["kappa_level", "kappa_slope", "phi_slope", "b_mult"]]

# Combine the filtered data into a new DataFrame
combined_df = DataFrame(
    Treino_Lognormal_kappa_level = [lognormal_additive[:,"kappa_level"][1], lognormal_nonlinear[:,"kappa_level"][1]],
    Treino_Lognormal_kappa_slope = [lognormal_additive[:,"kappa_slope"][1], lognormal_nonlinear[:,"kappa_slope"][1]],
    Treino_Lognormal_phi_slope = [lognormal_additive[:,"phi_slope"][1], lognormal_nonlinear[:,"phi_slope"][1]],
    Treino_Lognormal_phi = [lognormal_additive[:,"b_mult"][1], lognormal_nonlinear[:,"b_mult"][1]],
    Treino_Gamma_kappa_level = [gamma_additive[:,"kappa_level"][1], gamma_nonlinear[:,"kappa_level"][1]],
    Treino_Gamma_kappa_slope = [gamma_additive[:,"kappa_slope"][1], gamma_nonlinear[:,"kappa_slope"][1]],
    Treino_Gamma_phi_slope = [gamma_additive[:,"phi_slope"][1], gamma_nonlinear[:,"phi_slope"][1]],
    Treino_Gamma_phi = [gamma_additive[:,"b_mult"][1], gamma_nonlinear[:,"b_mult"][1]]
)

CSV.write("combined_df.csv", combined_df)

# Rename the rows to reflect the model type
combined_df[:Model] = ["additive", "non linear"]

# Reorder the columns to place 'Model' at the beginning
combined_df = combined_df[:, [:Model, :Treino_Lognormal_kappa_level, :Treino_Lognormal_kappa_slope, :Treino_Lognormal_phi_slope, :Treino_Lognormal_phi, :Treino_Gamma_kappa_level, :Treino_Gamma_kappa_slope, :Treino_Gamma_phi_slope, :Treino_Gamma_phi]]
