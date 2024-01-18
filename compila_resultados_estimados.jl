using CSV, Plots, DataFrames, HypothesisTests, Distributions


" Obtendo dados dos modelos estimados "

current_path = pwd()

path_saidas = current_path * "\\Saidas\\CombNaoLinear\\SazoDeterministica\\"

combinations = ["additive\\", "multiplicative1\\", "multiplicative2\\", "multiplicative3\\"]

distributions = ["LogNormal\\", "Gamma\\"]

series = ["ena", "carga", "uk_visits"]

df_fitted_values = CSV.read(path_saidas*combinations[1]*distributions[1]*"carga_fitted_values.csv", DataFrame)

dict_fit_in_sample = Dict()
dict_b = Dict()

for combination in combinations
    dict_fit_in_sample[combination] = Dict()
    dict_b[combination] = Dict()
    for distribution in distributions
        dict_fit_in_sample[combination][distribution] = Dict()
        dict_b[combination][distribution] = Dict()
        for serie in series
            # println(path_saidas*combination*distribution*"$(serie)_fitted_values.csv")
            df_fitted_values = CSV.read(path_saidas*combination*distribution*"$(serie)_fitted_values.csv", DataFrame)
            dict_fit_in_sample[combination][distribution][serie] = df_fitted_values[:,"fit_in_sample"]

            df_params = CSV.read(path_saidas*combination*distribution*"$(serie)_params.csv", DataFrame)
            distribution == "Gamma\\" ? param = "param_2_" : param = "param_1_"

            if combination == "multiplicative3\\"
                dict_b[combination][distribution][serie] = df_params[:, param*"b_mult"][1]
            end 
        end
    end
end


for serie in series
    for distribution in distributions
        fit_add = dict_fit_in_sample["additive\\"]["$(distribution)"][serie]
        fit_m1 = dict_fit_in_sample["multiplicative1\\"]["$(distribution)"][serie]
        fit_m2 = dict_fit_in_sample["multiplicative2\\"]["$(distribution)"][serie]
        fit_m3 = dict_fit_in_sample["multiplicative3\\"]["$(distribution)"][serie]

        p0 = plot(fit_add, fit_add, seriestype=:scatter, title="Fit in sample add X add", xlabel = "add", ylabel = "add", label="")
        p1 = plot(fit_add, fit_m1, seriestype=:scatter, title="Fit in sample add X mult1", xlabel = "add", ylabel = "mult1", label="")
        p2 = plot(fit_add, fit_m2, seriestype=:scatter, title="Fit in sample add X mult2", xlabel = "add", ylabel = "mult2", label="")
        p3 = plot(fit_add, fit_m3, seriestype=:scatter, title="Fit in sample add X mult3", xlabel = "add", ylabel = "mult3", label="")

        plot(p1,p2,p3, layout=grid(3,1), size=(800,800), plot_title = "$(distribution) - $(serie)")
        savefig("C:\\Users\\matno\\OneDrive\\Documentos\\PUC\\0_Períodos\\TCC\\Saidas\\Slides\\dispersao_fit_$(serie)_$(distribution[1:end-1]).png")

    end
end    


for distribution in distributions
    b = dict_b["multiplicative3\\"][distribution]

    println(distribution)
    println("carga     -> ",b["carga"])
    println("ena       -> ", b["ena"])
    println("uk_visits -> ", b["uk_visits"])
end