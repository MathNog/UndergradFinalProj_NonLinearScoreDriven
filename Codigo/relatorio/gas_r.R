library(GAS)

# Especificação do modelo
GASSpec = UniGASSpec(Dist = "gamma", ScalingType = "Identity",
                     GASPar = list(location = TRUE, scale = TRUE))
Fit       = UniGASFit(GASSpec, y) # estima o modelo
residuals = residuals(Fit) # obtém os resíduos do modelo estimado
forecast  = UniGASFor(Fit, 12) #gera a previsao 12 passos a frente