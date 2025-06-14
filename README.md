# Non-Gaussian Score-Driven Models with Non-Linear Unobserved Components Combination

## Author  
Matheus Nogueira  
📧 matnogueira@gmail.com

## Advisor  
Prof. Cristiano Fernandes

Department of Electrical Engineering  
Department of Informatics  
Pontifical Catholic University of Rio de Janeiro (PUC-Rio)

## Project Overview

This repository contains the materials related to my undergraduate thesis, which received a **maximum grade (10.0)** and was developed as part of the Computer Engineering program at **PUC-Rio**.

The project proposes and implements a new class of univariate **non-Gaussian Score-Driven (GAS)** models with **non-linear combinations of unobserved components**, such as trend and seasonality. It extends the framework implemented in the open-source Julia package [`UnobservedComponentsGAS`](https://github.com/LAMPSPUC/UnobservedComponentsGAS), developed by the LAMPS laboratory at PUC-Rio.

Key contributions include:
- Allowing **nonlinear interactions** among components (e.g., multiplicative trend-seasonality structures).
- Generalizing the model to **non-Gaussian conditional distributions** using the GAS updating mechanism.
- Estimating the models via **maximum likelihood** using robust optimization techniques.
- Applying the models to real-world time series and comparing their performance to traditional additive structures.

The final report can be accessed via the PUC-Rio Maxwell System:  
🔗 [PUC-Rio Maxwell Repository – Project #66255](https://www.maxwell.vrac.puc-rio.br/colecao.php?strSecao=resultado&nrSeq=66255&idi=1)

## Academic Output

This project led to a peer-reviewed paper presented at the **Brazilian Symposium on Operational Research (SBPO 2024)**:

📄 *Non-Gaussian Score-Driven Models with Non-Linear Unobserved Components Combination*  
📚 [Proceedings – SBPO 2024](https://proceedings.science/sbpo/sbpo-2024/trabalhos/non-gaussian-score-driven-models-with-non-linear-unobserved-components-combinati?lang=pt-br)

## Acknowledgments

This work was developed at the intersection of the Electrical Engineering and Informatics departments and benefited from discussions within the LAMPS research group.
