library(GAS)

# Especifica��o do modelo
GASSpec = UniGASSpec(Dist = "gamma", ScalingType = "Identity",
                     GASPar = list(location = TRUE, scale = TRUE))
Fit       = UniGASFit(GASSpec, y) # estima o modelo
residuals = residuals(Fit) # obt�m os res�duos do modelo estimado
forecast  = UniGASFor(Fit, 12) #gera a previsao 12 passos a frente