using DataFrames, GLM, Random, StatsBase

# using Pkg
# Pkg.add("RData")
using RData

data_path = "data/NorthCarolina_data.dat"

original_data = RData.load(data_path)

infants = original_data["infants"]

sim_data = infants[:, [:weeks, :sex, :race, :smoker, :mage, :weight]]

# Fit linear model
fitLinear = lm(@formula(weight ~ weeks + mage + sex + race), sim_data)

# Add mu_hat column to sim_data
sim_data.mu_hat = predict(fitLinear)

# Initialize betaHat and seHat matrices
R = 1000
betaHat = Matrix{Float64}(undef, R, 6)
seHat = Matrix{Float64}(undef, R, 6)

# Simulations
for r in 1:R
    sim_data.Y = sim_data.mu_hat + randn(nrow(sim_data)) .* ((sim_data.smoker .* 0.55) .+ (1 .- sim_data.smoker) .* 0.45)

    fitSim = lm(@formula(Y ~ weeks + sex + race + smoker + mage), sim_data)
    coef_summ = coeftable(fitSim)

    betaHat[r, :] .= coef(fitSim)
    seHat[r, :] .= stderror(fitSim)
end

# Print results
show(simRes, allcols=true, splitcols=false, alignment=:l)