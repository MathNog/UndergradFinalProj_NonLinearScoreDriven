using CSV, DataFrames, Plots

current_path = pwd()

path_series = current_path*"\\Dados\\Tratados\\"
path_saida = current_path*"\\Saidas\\Apresentacao\\"

ena              = CSV.read(path_series*"ena_limpo.csv",DataFrame)
carga            = CSV.read(path_series*"carga_limpo.csv", DataFrame)
uk_visits        = CSV.read(path_series*"uk_visits.csv", DataFrame)
precipitacao     = CSV.read(path_series*"uhe_belo_monte_limpo.csv", DataFrame)


dict_series                 = Dict()
dict_series["ena"]          = Dict()
dict_series["carga"]        = Dict()
dict_series["viagens"]    = Dict()
dict_series["precipitacao"]    = Dict()

dict_series["ena"]["values"]   = ena[:,:ENA]
dict_series["ena"]["dates"]    = ena[:,:Data]

dict_series["carga"]["values"] = carga[:,:Carga]
dict_series["carga"]["dates"]  = carga[:,:Data]

dict_series["viagens"]["values"] = uk_visits[:,:Valor]
dict_series["viagens"]["dates"]  = uk_visits[:,:Data]

dict_series["precipitacao"]["values"] = precipitacao[:,:MEDIA]
dict_series["precipitacao"]["dates"]  = precipitacao[:,:DATA]

names_series = Dict("carga"=>"carga", "ena"=>"ena", "viagens"=>"viagens", "precipitacao"=>"precipitacao")

for serie in ["ena", "carga", "viagens", "precipitacao"]

    titulo = "Série de $(names_series[serie])"
    y = dict_series[serie]["values"]
    d = dict_series[serie]["dates"]
    if serie in ["ena", "precipitacao"]
        titulo *= " e seu histograma"

        p1 = plot(d, y, label= "")
        p1 = vline!([d[end-11]], color="red", ls=:dash, label="Separação treino e teste")
        h = histogram(y, label="")
        plot(p1, h, layout=grid(2,1), size=(900,500), plot_title = titulo)
        savefig(path_saida*"$serie.png")
    else
        p1 = plot(d, y, label= "")
        p1 = vline!([d[end-11]], color="red", ls=:dash, label="Separação treino e teste")
        plot(p1, layout=grid(1,1), size=(600,200), plot_title = titulo)
        savefig(path_saida*"$serie.png")
    end
end

