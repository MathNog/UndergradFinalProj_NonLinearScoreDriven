using Plots, Distributions, StatsPlots, StatsBase, Statistics, SpecialFunctions,LinearAlgebra, JSON
using JSON3, Random

function score_normal(μ, σ², y) 
    println("score_normal")
    return [(y - μ)/σ²; -(0.5/σ²) * (1 - ((y - μ)^2)/σ²)]
end

function fisher_information_normal(μ, σ²)
    println("fisher_information_normal")
    return [1/(σ²) 0; 0 1/(2*(σ²^2))]
end


function Γ(x)
    return SpecialFunctions.gamma(x)
end
    

"Auxiliar function Ψ1(α)"
function Ψ1(α)
    return SpecialFunctions.digamma(α)
end


"Auxiliar function Ψ2(α)"
function Ψ2(α)
    return SpecialFunctions.trigamma(α)
end


"
Evaluate the score of a Normal distribution with mean μ and variance σ², in observation y.
"
function score_gama(α, λ, y) 
  
    α <= 0 ? α = 1e-2 : nothing
    λ <= 0 ? λ = 1e-4 : nothing

    # ∇_α =  log(y) - y/λ + log(α) - Ψ1(α) - log(λ) + 1
    # ∇_λ = (α/λ)*((y/λ)-1)

    ∇_α_exp = α * (log(y) - y/λ + log(α) - Ψ1(α) - log(λ) + 1)
    ∇_λ_exp = α * (y/λ - 1)
    
    return [∇_α_exp; ∇_λ_exp]
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
Colocar link aqui
"
function fisher_information_gama(α, λ) 

    α <= 0 ? α = 1e-2 : nothing
    λ <= 0 ? λ = 1e-4 : nothing
    
    # I_λ = α/(λ^2)
    # I_α = Ψ2(α) - 1/α

    I_α_exp = α^2 * (Ψ2(α) - 1/α)
    I_λ_exp = α

    return [I_α_exp 0; 0 I_λ_exp]
end


const DICT_SCORE = Dict("Normal" => score_normal,
                        "Gamma"  => score_gama)

const DICT_FISHER_INFORMATION = Dict("Normal" => fisher_information_normal,
                                    "Gamma"   => fisher_information_gama)

function scaled_score(dist_name, first_param, second_param, y, d)
    ∇ = DICT_SCORE[dist_name](first_param, second_param, y)
    println("Saiu score")
    if d == 0.0 
        s = Matrix(I, length(∇), length(∇))' * ∇

    elseif d == 0.5
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param)
        s = cholesky(inv(FI), check = false).UL' * ∇

    else
        FI = DICT_FISHER_INFORMATION[dist_name](first_param, second_param)
        s = inv(FI) * ∇
    end
    if dist_name == "Normal"
        return s[1]
    else
        return s[2]
    end
end

function plot_ACF(y)
    acf_values = autocor(y)
    lag_values = collect(0:length(acf_values) - 1)
    conf_interval = 1.96 / sqrt(length(y))  # 95% confidence interval
    plot(title="FAC")
    plot!(autocor(y),seriestype=:stem, label="")
    hline!([conf_interval, -conf_interval], line = (:red, :dash), label = "IC 95%")
end



current_path = pwd()
path_dados = current_path*"/Dados/SeriesArtificiais/"

T = 200
# combination = "additive"
combination = "multiplicative1"
combination = "multiplicative2"

if combination == "additive"
    dict_inicial_lognormal_ar  = JSON3.read(path_dados*"fitted_lognormal_ar_sazo.json")
    dict_inicial_lognormal_rws = JSON3.read(path_dados*"fitted_lognormal_rws_sazo.json")
    dict_inicial_gamma_ar      = JSON3.read(path_dados*"fitted_gamma_ar_sazo.json")
    dict_inicial_gamma_rws     = JSON3.read(path_dados*"fitted_gamma_rws_sazo.json")
elseif combination == "multiplicative1"
    dict_inicial_lognormal_ar  = JSON3.read(path_dados*"fitted_lognormal_ar_sazo_multiplicative1.json")
    dict_inicial_lognormal_rws = JSON3.read(path_dados*"fitted_lognormal_rws_sazo_multiplicative1.json")
    dict_inicial_gamma_ar      = JSON3.read(path_dados*"fitted_gamma_ar_sazo_multiplicative1.json")
    dict_inicial_gamma_rws     = JSON3.read(path_dados*"fitted_gamma_rws_sazo_multiplicative1.json")
elseif combination == "multiplicative2"
    dict_inicial_lognormal_ar  = JSON3.read(path_dados*"fitted_lognormal_ar_sazo_multiplicative2.json")
    dict_inicial_lognormal_rws = JSON3.read(path_dados*"fitted_lognormal_rws_sazo_multiplicative2.json")
    dict_inicial_gamma_ar      = JSON3.read(path_dados*"fitted_gamma_ar_sazo_multiplicative2.json")
    dict_inicial_gamma_rws     = JSON3.read(path_dados*"fitted_gamma_rws_sazo_multiplicative2.json")
end


" --------------------- Lognormal RWS com Sazo -------------------------"

path_saida = current_path*"\\Saidas\\SeriesArtificiais\\$combination\\"

for j in 1:5
    RWS    = Vector{Float64}(undef, T+1)
    b      = Vector{Float64}(undef, T+1)
    S      = Vector{Float64}(undef, T+1)
    μ      = Vector{Float64}(undef, T+1)
    σ      = Vector{Float64}(undef, T+1)
    scores = Vector{Float64}(undef, T+1)
    γ      = [zeros(6) for i in 1:T+1]
    γ_star = [zeros(6) for i in 1:T+1]
    S      = Vector{Float64}(undef, T+1)
    y      = Vector{Float64}(undef, T)

    ϕb    = 1.#dict_inicial_lognormal_rws[:slope][:hyperparameters][:ϕb]
    c     = dict_inicial_lognormal_rws[:intercept]
    κ_RWS = dict_inicial_lognormal_rws[:level][:hyperparameters][:κ]
    κ_b   = dict_inicial_lognormal_rws[:slope][:hyperparameters][:κ]
    κ_S   = dict_inicial_lognormal_rws[:seasonality][:hyperparameters][:κ]

    γ[1]      = dict_inicial_lognormal_rws[:seasonality][:hyperparameters][:γ][7:12]
    γ_star[1] = dict_inicial_lognormal_rws[:seasonality][:hyperparameters][:γ_star][7:12]
    S[1]      = dict_inicial_lognormal_rws[:seasonality][:value][2]
    RWS[1]    = dict_inicial_lognormal_rws[:level][:value][2]
    b[1]      = dict_inicial_lognormal_rws[:slope][:value][2]
    μ[1]      = dict_inicial_lognormal_rws[:param_1]
    σ[1]      = 1#dict_inicial_lognormal_rws[:param_2]

    d = 0.
    num_harmonics   = 6
    seasonal_period = 12

    for t in 1:T
        @info "t = $t"
        # Sorteia um y com os parametros correntes
        y[t] = rand(Normal(μ[t],σ[t])) #Mudar para lognormal
        
        #Atualiza o score com o novo y
        # μ_aux = log(μ[t]) - 0.5*log(1+(σ[t]^2)/μ[t]^2)
        # σ_aux = log(1 + (σ[t]^2/μ[t]^2))

        s = scaled_score("Normal", μ[t], σ[t]^2, y[t], d)
        # Atualiza os parametros a partir da dinamica
        b[t+1]   = ϕb*b[t] + κ_b*s
        RWS[t+1] = RWS[t] + b[t] + κ_RWS*s

        for i in 1:num_harmonics
            γ[t+1][i]      = γ[t][i] * cos(2*π*i / seasonal_period) + γ_star[t][i] * sin(2*π*i / seasonal_period) + κ_S*s
            γ_star[t+1][i] = -γ[t][i] * sin(2*π*i / seasonal_period) + γ_star[t][i] * cos(2*π*i / seasonal_period) + κ_S*s
        end

        S[t+1] = sum(γ[t+1])

        if combination == "additive"
            μ[t+1] = c + RWS[t+1] + S[t+1]
        elseif combination == "multiplicative1"
            μ[t+1] = c + (RWS[t+1] * S[t+1])
        elseif combination == "multiplicative2"
            μ[t+1] = c + RWS[t+1] *(1 + S[t+1])
        end

        σ[t+1] = σ[t]
        println("    s = $(s) | b = $(b[t]) | RWS = $(RWS[t]) | S = $(S[t]) | μ = $(μ[t]) | σ = $(σ[t])")
    end

    y_log = y .+ 2*abs(minimum(y))
    y_log = log.(y_log)

    if combination == "additive"
        title = "Série $j - RWS + Sazo - LogNormal"
    elseif combination == "multiplicative1"
        title = "Série $j - RWS × Sazo - LogNormal"
    elseif combination == "multiplicative2"
        title = "Série $j - RWS × (1 + Sazo) - LogNormal"
    end

    p1 = plot(y_log[2:end],label="")
    p2 = plot_ACF(diff(y_log[2:end]))
    plot(p1, p2,  layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_rws_lognormal_serie_$(j).png")

    p_RWS = plot(RWS[2:end],label="Level")
    p_sazo  = plot(S[2:end],label="Seasonality")
    plot(p_RWS, p_sazo, layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_rws_lognormal_components_$(j).png")
end


" --------------------- Lognormal AR com Sazo -------------------------"

path_saida = current_path*"\\Saidas\\SeriesArtificiais\\$combination\\"
num_harmonics   = 6
seasonal_period = 12
for j in 1:5
    AR     = Vector{Float64}(undef, T+1)
    S      = Vector{Float64}(undef, T+1)
    μ      = Vector{Float64}(undef, T+1)
    σ      = ones(T+1)#Vector{Float64}(undef, T)
    scores = Vector{Float64}(undef, T+1)
    γ      = [zeros(6) for i in 1:T+1]
    γ_star = [zeros(6) for i in 1:T+1]
    S      = Vector{Float64}(undef, T+1)
    y      = Vector{Float64}(undef, T)

    c    = dict_inicial_lognormal_ar[:intercept]
    ϕ    = dict_inicial_lognormal_ar[:ar][:hyperparameters][:ϕ][1]
    κ_AR = dict_inicial_lognormal_ar[:ar][:hyperparameters][:κ][1]
    κ_S  = 10*dict_inicial_lognormal_ar[:seasonality][:hyperparameters][:κ][1]
    d    = 0.

    AR[1]     = dict_inicial_lognormal_ar[:ar][:value][2]
    γ[1]      = dict_inicial_lognormal_ar[:seasonality][:hyperparameters][:γ][7:12]
    γ_star[1] = dict_inicial_lognormal_ar[:seasonality][:hyperparameters][:γ_star][7:12]
    S[1]      = dict_inicial_lognormal_ar[:seasonality][:value][2]
    σ[1]      = dict_inicial_lognormal_ar[:param_2]
    μ[1]      = dict_inicial_lognormal_ar[:param_1]

    for t in 1:T
        @info "t = $t"
        # Sorteia um y com os parametros correntes
        y[t] = rand(Normal(μ[t],σ[t]^2))
        #Atualiza o score com o novo y
        s = scaled_score("Normal", μ[t], σ[t]^2, y[t], d)
        # Atualiza os parametros a partir da dinamica
        AR[t+1] = ϕ*AR[t] .+ κ_AR*s

        for i in 1:num_harmonics
            γ[t+1][i]      = γ[t][i] * cos(2*π*i / seasonal_period) + γ_star[t][i] * sin(2*π*i / seasonal_period) + κ_S*s
            γ_star[t+1][i] = -γ[t][i] * sin(2*π*i / seasonal_period) + γ_star[t][i] * cos(2*π*i / seasonal_period) + κ_S*s
        end

        S[t+1] = sum(γ[t+1])

        if combination == "additive"
            μ[t+1] = c + AR[t+1] + S[t+1]
        elseif combination == "multiplicative1"
            μ[t+1] = c + (AR[t+1] * S[t+1])
        elseif combination == "multiplicative2"
            μ[t+1] = c + AR[t+1] * (1 + S[t+1])
        end

        σ[t+1] = σ[t]

        println("    s = $(s) | AR = $(AR[t]) | S = $(S[t]) | μ = $(μ[t]) | σ = $(σ[t])")
    end

    y_log = y .+ abs(minimum(y)) .+ 1.
    y_log = log.(y_log)

    if combination == "additive"
        title = "Série $j - AR(1) + Sazo - LogNormal"
    elseif combination == "multiplicative1"
        title = "Série $j - AR(1) × Sazo - LogNormal"
    elseif combination == "multiplicative2"
        title = "Série $j - AR(1) × (1 + Sazo) - LogNormal"
    end

    p1 = plot(y_log,label="")
    p2 = plot_ACF(y_log)
    plot(p1, p2,  layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_ar_lognormal_serie_$(j).png")

    p_AR = plot(AR[2:end],label="AR")
    p_sazo  = plot(S[2:end],label="Seasonality")
    plot(p_AR, p_sazo, layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_ar_lognormal_components_$(j).png")
end

" ---------------------- Gamma RWS com Sazo---------------------------------"

path_saida = current_path*"\\Saidas\\SeriesArtificiais\\$combination\\"

for j in 1:5
    λ      = Vector{Float64}(undef, T+1)
    α      = ones(T+1)#Vector{Float64}(undef, T)
    RWS    = Vector{Float64}(undef, T+1)
    b      = Vector{Float64}(undef, T+1)
    S      = Vector{Float64}(undef, T+1)
    scores = Vector{Float64}(undef, T+1)
    γ      = [zeros(6) for i in 1:T+1]
    γ_star = [zeros(6) for i in 1:T+1]
    S      = Vector{Float64}(undef, T+1)
    y      = Vector{Float64}(undef, T)

    ϕb    = 0.9#dict_inicial_gamma_rws[:slope][:hyperparameters][:ϕb]
    c     = dict_inicial_gamma_rws[:intercept][:values][1]
    κ_RWS = dict_inicial_gamma_rws[:rws][:κ]
    κ_b   = dict_inicial_gamma_rws[:slope][:κ]
    κ_S   = dict_inicial_gamma_rws[:seasonality][:κ] 

    γ[1]      = dict_inicial_gamma_rws[:seasonality][:γ][1] 
    γ_star[1] = dict_inicial_gamma_rws[:seasonality][:γ_star][1] 
    S[1]      = dict_inicial_gamma_rws[:seasonality][:values][1] 
    RWS[1]    = dict_inicial_gamma_rws[:rws][:values][2] 
    b[1]      = dict_inicial_gamma_rws[:slope][:values][2] 
    α[1]      = dict_inicial_gamma_rws[:fixed_param][1]
    λ[1]      = exp.(dict_inicial_gamma_rws[:param][1][1])

    d = 1.0
    num_harmonics   = 6
    seasonal_period = 12
    
    for t in 1:T
        @info "t = $t"
        # Sorteia um y com os parametros correntes
        y[t] = rand(Gamma(α[t],λ[t]/α[t]))
        #Atualiza o score com o novo y
        s = scaled_score("Gamma", α[t], λ[t], y[t], d)
        # Atualiza os parametros a partir da dinamica
        b[t+1]   = ϕb*b[t] + κ_b*s
        RWS[t+1] = RWS[t] + b[t] + κ_RWS*s

        for i in 1:num_harmonics
            γ[t+1][i]      = γ[t][i] * cos(2*π*i / seasonal_period) + γ_star[t][i] * sin(2*π*i / seasonal_period) + κ_S*s
            γ_star[t+1][i] = -γ[t][i] * sin(2*π*i / seasonal_period) + γ_star[t][i] * cos(2*π*i / seasonal_period) + κ_S*s
        end

        S[t+1] = sum(γ[t+1])

        if combination == "additive"
            λ[t+1] = exp.(c + (RWS[t+1] + S[t+1]))
        elseif combination == "multiplicative1"
            λ[t+1] = exp.(c + (RWS[t+1] * S[t+1]))
        else
            λ[t+1] = exp.(c + RWS[t+1] * (1+ S[t+1]))
        end
        α[t+1] = α[t]

        println("    s = $(s) | b = $(b[t]) | RWS = $(RWS[t]) | S = $(S[t]) | λ = $(λ[t]) | α = $(α[t])")
    end

    if combination == "additive"
        title = "Série $j - RWS + Sazo - Gamma"
    elseif combination == "multiplicative1"
        title = "Série $j - RWS × Sazo - Gamma"
    elseif combination == "multiplicative2"
        title = "Série $j - RWS × (1 + Sazo) - Gamma"
    end

    p1 = plot(y,label="")
    p2 = plot_ACF(diff(y))
    plot(p1, p2,  layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_rws_gamma_serie_$(j).png")

    p_RWS = plot(RWS[2:end],label="Level")
    p_sazo  = plot(S[2:end],label="Seasonality")
    plot(p_RWS, p_sazo, layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_rws_gamma_components_$(j).png")
end

" ---------------------- Gamma AR com Sazo---------------------------------"

path_saida = current_path*"\\Saidas\\SeriesArtificiais\\$combination\\"

for j in 1:5
    AR     = Vector{Float64}(undef, T+1)
    S      = Vector{Float64}(undef, T+1)
    λ      = Vector{Float64}(undef, T+1)
    α      = ones(T+1)#Vector{Float64}(undef, T)
    scores = Vector{Float64}(undef, T+1)
    γ      = [zeros(6) for i in 1:T+1]
    γ_star = [zeros(6) for i in 1:T+1]
    S      = Vector{Float64}(undef, T+1)
    y      = Vector{Float64}(undef, T)

    c    = dict_inicial_gamma_ar[:intercept][:values][1]
    ϕ    = 0.9#dict_inicial_gamma_ar[:ar][:ϕ][1]
    κ_AR = dict_inicial_gamma_ar[:ar][:κ]
    κ_S  = dict_inicial_gamma_ar[:seasonality][:κ]
    d    = 1.

    AR[1]     = dict_inicial_gamma_ar[:ar][:values][2]
    γ[1]      = dict_inicial_gamma_ar[:seasonality][:γ][1]
    γ_star[1] = dict_inicial_gamma_ar[:seasonality][:γ_star][1]
    S[1]      = dict_inicial_gamma_ar[:seasonality][:values][2]
    λ[1]      = exp(dict_inicial_gamma_ar[:param][1])
    α[1]      = dict_inicial_gamma_ar[:fixed_param][1]

    for t in 1:T
        @info "t = $t"
        # Sorteia um y com os parametros correntes
        y[t] = rand(Gamma(α[t],λ[t]/α[t]))
        #Atualiza o score com o novo y
        s = scaled_score("Gamma", α[t], λ[t], y[t], d)
        # Atualiza os parametros a partir da dinamica
        AR[t+1] = ϕ*AR[t] + κ_AR*s

        for i in 1:num_harmonics
            γ[t+1][i]      = γ[t][i] * cos(2*π*i / seasonal_period) + γ_star[t][i] * sin(2*π*i / seasonal_period) + κ_S*s
            γ_star[t+1][i] = -γ[t][i] * sin(2*π*i / seasonal_period) + γ_star[t][i] * cos(2*π*i / seasonal_period) + κ_S*s
        end

        S[t+1] = sum(γ[t+1])

        if combination == "additive"
            λ[t+1] = exp(c + AR[t+1] + S[t+1])
        elseif combination == "multiplicative1"
            λ[t+1] = exp(c + (AR[t+1] * S[t+1]))
        else
            λ[t+1] = exp.(c + AR[t+1] * (1+ S[t+1]))
        end
        α[t+1] = α[t]

        println("    s = $(s) | AR = $(AR[t]) | S = $(S[t]) | λ = $(λ[t]) | α = $(α[t])")
    end

    if combination == "additive"
        title = "Série $j - AR(1) + Sazo - Gamma"
    elseif combination == "multiplicative1"
        title = "Série $j - AR(1) × Sazo - Gamma"
    elseif combination == "multiplicative2"
        title = "Série $j - AR(1) × (1 + Sazo) - Gamma"
    end

    p1 = plot(y[2:end],label="")
    p2 = plot_ACF(y[2:end])
    plot(p1, p2,  layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_ar_gamma_serie_$(j).png")

    p_AR = plot(AR[2:end],label="AR")
    p_sazo  = plot(S[2:end],label="Seasonality")
    plot(p_AR, p_sazo, layout=grid(2,1), size=(800,600), plot_title = title)
    savefig(path_saida*"$(combination)_ar_gamma_components_$(j).png")
end

