
" AutoARIMA Benchmark"

# path_saida = current_path*"\\Saidas\\Benchmark\\2parametros\\AutoARIMA\\"

# dict_benchmarks          = Dict()
# dict_benchmarks["carga"] = Dict()
# dict_benchmarks["ena"]   = Dict()
# dict_benchmarks["vazao"] = Dict()

# dict_benchmarks["carga"]["fit"]      = CSV.read(path_series*"carga_fit_autoarima.csv",DataFrame)
# dict_benchmarks["carga"]["forecast"] = CSV.read(path_series*"carga_forecast_autoarima.csv",DataFrame)

# dict_benchmarks["ena"]["fit"]      = CSV.read(path_series*"ena_fit_autoarima.csv",DataFrame)
# dict_benchmarks["ena"]["forecast"] = CSV.read(path_series*"ena_forecast_autoarima.csv",DataFrame)

# dict_benchmarks["vazao"]["fit"]      = CSV.read(path_series*"vazao_fit_autoarima.csv",DataFrame)
# dict_benchmarks["vazao"]["forecast"] = CSV.read(path_series*"vazao_forecast_autoarima.csv",DataFrame)


# for serie in ["carga", "ena", "vazao"]
    
#     y = dict_series[serie]["values"]
#     dates = dict_series[serie]["dates"]
#     fit = dict_benchmarks[serie]["fit"].Values
#     dates_fit = dict_benchmarks[serie]["fit"].Data
#     prev = dict_benchmarks[serie]["forecast"].Values
#     dates_prev = dict_benchmarks[serie]["forecast"].Data

#     residuals = y[1:end-12] .- fit
#     std_residuals = (residuals .- mean(residuals))./std(residuals)

#     mape_train = round(100*MAPE(y[1:end-12],fit), digits=2)
#     mape_test = round(100*MAPE(y[end-11:end],prev), digits=2)
#     df_mapes = DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
#     CSV.write(path_saida*"$(serie)_mapes.csv",df_mapes)

#     plot(title="Fit in sample AutoARIMA $serie")
#     plot!(dates, y, label="Série")
#     plot!(dates_fit, fit, label="Fit: MAPE = $mape_train%")
#     savefig(path_saida*"$(serie)_fit_autoarima.png")

#     plot(title="Forecast AutoARIMA $serie")
#     plot!(dates[end-11:end], y[end-11:end], label="Série")
#     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
#     savefig(path_saida*"$(serie)_forecast_autoarima.png")

#     plot(title="Fit and Forecast AutoARIMA $serie")
#     plot!(dates, y, label="Série")
#     plot!(dates_fit, fit, label="Fit; MAPE = $mape_train%")
#     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
#     savefig(path_saida*"$(serie)_fit_forecast_autoarima.png")

#     qq = plot(qqplot(Normal, std_residuals))
#     qq = plot!(title="QQPlot Residuos")
#     h = histogram(std_residuals, title="Histograma Residuos", label="")
#     r = plot(title="Resíduos Padronizados")
#     r = plot!(dates_fit, std_residuals , label="Resíduos")
#     acf_values = autocor(std_residuals)
#     lag_values = collect(0:length(acf_values) - 1)
#     conf_interval = 1.96 / sqrt(length(residuals)-1)  # 95% confidence interval
#     a = plot(title="FAC dos Residuos")
#     a = plot!(autocor(std_residuals),seriestype=:stem, label="")
#     a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
#     plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
#         plot_title = "Diagnosticos Residuos AutoARIMA - $serie", title=["Resíduos Padronizados" "FAC dos Residuos" "Histograma Residuos" "QQPlot Residuos"])
#     savefig(path_saida*"$(serie)_diagnosticos_autoarima.png")    
# end

