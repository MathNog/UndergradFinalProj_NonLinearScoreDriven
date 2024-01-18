using CSV, DataFrames, Plots

current_path = pwd()

path_series = current_path*"\\Dados\\Tratados\\"
path_saida = current_path*"\\Saidas\\Relatorio\\SeriesTestes\\"

ena              = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga            = CSV.read(path_series*"carga_limpo.csv", DataFrame)
uk_visits        = CSV.read(path_series*"uk_visits.csv", DataFrame)


dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["uk_visits"]    = Dict()

dict_series["ena"]["values"]   = ena[:,:ENA]
dict_series["ena"]["dates"]    = ena[:,:Data]

dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"]  = carga[:,:Data]

dict_series["uk_visits"]["values"] = uk_visits[:,:Valor]
dict_series["uk_visits"]["dates"]  = uk_visits[:,:Data]



for serie in ["ena", "carga", "uk_visits"]

    titulo = "Série de $serie"
    y = dict_series[serie]["values"]
    d = dict_series[serie]["dates"]
    if serie == "ena"
        titulo *= " e seu histograma"

        p1 = plot(d, y, label= "")
        h = histogram(y, label="")
        plot(p1, h, layout=grid(2,1), size=(900,500), plot_title = titulo)
        savefig(path_saida*"$serie.png")
    else
        p1 = plot(d, y, label= "")
        plot(p1, layout=grid(1,1), size=(600,200), plot_title = titulo)
        savefig(path_saida*"$serie.png")
    end
end