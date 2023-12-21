
 "AutoARIMA Benchmark"
using CSV, DataFrames
using Plots, StatsBase, MLJBase

function MAPE(y_true, y_prev)
    return MLJBase.mape(y_prev, y_true)
end



path_series = current_path*"\\Dados\\Tratados\\"
path_saida = current_path*"\\Saidas\\Benchmark\\AutoARIMA\\"


ena          = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga        = CSV.read(path_series*"carga_limpo.csv", DataFrame)
carga_marina = CSV.read(path_series*"dados_cris.csv EMT_rural_cativo.csv", DataFrame)
airline      = CSV.read(path_series*"AirPassengers.csv", DataFrame)

dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["carga_marina"] = Dict()
dict_series["airline"]      = Dict()

dict_series["ena"]["values"]   = ena[:,:ENA]
dict_series["ena"]["dates"]    = ena[:,:Data]
dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"]  = carga[:,:Data]


dict_benchmarks          = Dict()
dict_benchmarks["carga"] = Dict()
dict_benchmarks["ena"]   = Dict()
dict_benchmarks["vazao"] = Dict()
dict_benchmarks["carga"]["fit"]      = CSV.read(path_series*"carga_fit_autoarima.csv",DataFrame)
dict_benchmarks["carga"]["forecast"] = CSV.read(path_series*"carga_forecast_autoarima.csv",DataFrame)
dict_benchmarks["ena"]["fit"]      = CSV.read(path_series*"ena_fit_autoarima.csv",DataFrame)
dict_benchmarks["ena"]["forecast"] = CSV.read(path_series*"ena_forecast_autoarima.csv",DataFrame)


 for serie in ["carga", "ena"]
   
     y = dict_series[serie]["values"]
     dates = dict_series[serie]["dates"]
     fit_values = dict_benchmarks[serie]["fit"].Values
     dates_fit = dict_benchmarks[serie]["fit"].Data
     prev = dict_benchmarks[serie]["forecast"].Values
     dates_prev = dict_benchmarks[serie]["forecast"].Data
     resid = y[1:end-12] .- fit_values
     std_resid = (resid .- mean(resid))./std(resid)
     mape_train = round(100*MAPE(y[1:end-12],fit_values), digits=2)
     mape_test = round(100*MAPE(y[end-11:end],prev), digits=2)
     df_mapes = DataFrame(Dict("MAPE Treino"=>mape_train, "MAPE Teste"=>mape_test))
     CSV.write(path_saida*"$(serie)_mapes.csv",df_mapes)

     plot(title="Fit in sample AutoARIMA $serie")
     plot!(dates, y, label="Série")
     plot!(dates_fit, fit_values, label="Fit: MAPE = $mape_train%")
     savefig(path_saida*"$(serie)_fit_autoarima.png")

     plot(title="Forecast AutoARIMA $serie")
     plot!(dates[end-11:end], y[end-11:end], label="Série")
     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
     savefig(path_saida*"$(serie)_forecast_autoarima.png")

     plot(title="Fit and Forecast AutoARIMA $serie")
     plot!(dates, y, label="Série")
     plot!(dates_fit, fit_values, label="Fit; MAPE = $mape_train%")
     plot!(dates_prev, prev, color="red", label="Previsão: MAPE = $mape_test%")
     savefig(path_saida*"$(serie)_fit_forecast_autoarima.png")

     plot(qqplot(Normal, std_resid))
     plot!(title="QQPlot Residuos AutoARIMA")
     savefig(path_saida*"$(serie)_qqplot_autoarima.png") 

     histogram(std_resid, title="Histograma Residuos AutoARIMA", label="")
     savefig(path_saida*"$(serie)_histograma_autoarima.png") 

     plot(title="Resíduos Padronizados AutoARIMA")
     plot!(dates_fit, std_resid , label="Resíduos")
     savefig(path_saida*"$(serie)_residuos_autoarima.png") 
     
     acf_values = autocor(std_resid)
     lag_values = collect(0:length(acf_values) - 1)
     conf_interval = 1.96 / sqrt(length(resid)-1)  # 95% confidence interval
     plot(title="FAC dos Residuos AutoARIMA")
     plot!(autocor(std_resid),seriestype=:stem, label="")
     hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
     savefig(path_saida*"$(serie)_acf_autoarima.png") 

     qq = plot(qqplot(Normal, std_resid))
     qq = plot!(title="QQPlot Residuos")
     h = histogram(std_resid, title="Histograma Residuos", label="")
     r = plot(title="Resíduos Padronizados")
     r = plot!(dates_fit, std_resid , label="Resíduos")
     acf_values = autocor(std_resid)
     lag_values = collect(0:length(acf_values) - 1)
     conf_interval = 1.96 / sqrt(length(resid)-1)  # 95% confidence interval
     a = plot(title="FAC dos Residuos")
     a = plot!(autocor(std_resid),seriestype=:stem, label="")
     a = hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
     plot(r, a, h, qq,  layout=grid(2,2), size=(1200,800), 
         plot_title = "Diagnosticos Residuos AutoARIMA - $serie", title=["Resíduos Padronizados" "FAC dos Residuos" "Histograma Residuos" "QQPlot Residuos"])
     savefig(path_saida*"$(serie)_diagnosticos_autoarima.png")    
 end

