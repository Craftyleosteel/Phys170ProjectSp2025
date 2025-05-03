### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ d0c11460-6e24-4960-9e3c-c67c572272b4
begin
	using Flux
	using Zygote
	using LinearAlgebra
	using Random
	using Distributions
	using Plots
	using Statistics
	using ProgressMeter
	using ThreadsX
	using CUDA
end

# ╔═╡ 1f58cc07-0388-4d62-a73d-5b352227111a
md"""
# CGNet In Julia
"""

# ╔═╡ d0fd2080-1985-11f0-00c2-bbfce2ad649b
md"""
Created By Ananya Venkatachalam and Paco Navarro
"""

# ╔═╡ 48bd4aa5-a6a5-4319-a6cb-821eebacc830
md"""
## First import relevant packages
"""

# ╔═╡ 0fc0db9d-f5f6-439d-bfee-475e20d8a3ee
md"""
# Define a toy potential for a proof of concept
"""

# ╔═╡ 7a63dbc2-c063-4866-a961-58ea3d594338
function V(x, y) #Toy potential from paper
	xterm = (x - 4)*(x + 2)*(x - 2)*(x + 3)
	ripples = (1/25) * sin(3 * (x + 5) * (x + 6))
	return (1/50) * xterm + 0.5 * y^2 + ripples
end

# ╔═╡ 6ed893e2-8186-4870-a22f-0da60934ff13
md"""
## Define The Simulation Dynamics
"""

# ╔═╡ 0a67a981-8dbc-435f-a5b0-0a50485ea676
begin
function simulate_trajectory(T, τ, D, seed)
	"""Simulates the trajectory for T particles in a potential"""
	Random.seed!(seed)
	x, y = 0.0, 0.0
	traj = zeros(T, 2)
	for t in 1:T
		∂Vx = Zygote.gradient(z -> V(z, y), x)[1]
		∂Vy = Zygote.gradient(z -> V(x, z), y)[1]
		x -= τ * ∂Vx + sqrt(2 * D * τ) * randn()
		y -= τ * ∂Vy + sqrt(2 * D * τ) * randn()
		traj[t, :] .= (x, y)
	end
	return traj
end

#High level threading implementation
all_trajs = ThreadsX.map(i -> simulate_trajectory(10_000, 0.01, 1.0, 42 + i), 1:100) 
traj = vcat(all_trajs...)
end

# ╔═╡ ebe02d8f-b03a-4535-a695-7620e95ac598
md"""
## Define How We are Computing the Force fields
"""

# ╔═╡ b5610bfd-c1b1-438c-bbad-074e45bc1578
begin
function compute_instantaneous_forces(traj)
	forces = []
	xs = []
	for (x, y) in eachrow(traj)
		Fx = -Zygote.gradient(z -> V(z, y), x)[1]
		push!(forces, Fx)
		push!(xs, x)
	end
	return hcat(xs...), hcat(forces...)
end

x_vals, fx_vals = compute_instantaneous_forces(traj)
end

# ╔═╡ 9e313963-c0ff-447e-86d1-9d96d503d5f5
md"""
## Define the Neural Network
"""

# ╔═╡ 5cc05899-400f-4baf-a983-2d85cdac14d8
md"""
## Define The Loss Function
"""

# ╔═╡ 2880942e-288a-4fd3-9bc6-b20be2b9593d
md"""
## Train the NN
"""

# ╔═╡ 9d72129f-06ae-4b5f-9074-399094da26e5
# Sample a subset of the data and convert to Float32
begin
	sample_inds = rand(1:length(x_vals), 5000)
	x_train = Float32.(x_vals[sample_inds])
	f_train = Float32.(fx_vals[sample_inds])
end


# ╔═╡ 3746a4e2-62c3-4473-b7bc-eab0b07e42e5
begin
CGnet = Chain(
	Dense(1, 20, relu),
	Dense(20, 20, relu),
	Dense(20, 1)
)
# Move model to GPU
CGnet = gpu(CGnet)

# Move training data to GPU once
x_train_gpu = gpu(reshape(Float32.(x_train), :, 1))  # (batchsize, 1)
f_train_gpu = gpu(Float32.(f_train))                 # (batchsize,)
end


#Smaller and simpler than the NN in the paper to help it train faster because we are working with a toy model. For a full implementation you may want to use this NN below: 
#
#CGnet = Chain(
#	Dense(1, 50, tanh),
#	Dense(50, 50, tanh),
#	Dense(50, 1) # output is scalar free energy
#)

# ╔═╡ 5552f664-c1bb-4c83-89aa-dcea9a5fce12
# Computes the mean squared error between predicted and true forces
function batch_loss_fast_gpu(xb, fb)
	# Forward pass: predict energy
	y = CGnet(xb)[:, 1]

	# Backward pass: compute gradient ∂U/∂x using Zygote
	grads = Zygote.gradient(() -> sum(y)) do
		y = CGnet(xb)[:, 1]
		sum(y)
	end

	# Predicted force is negative energy gradient
	f_pred = -grads[1]

	# Return mean squared error between predicted and true forces
	return mean((f_pred .- fb).^2)
end

# ╔═╡ 65a0e2c6-4e1c-4ed4-bec6-2c428d9b95df
begin
	opt = Optimisers.Adam(0.01)
	state = Optimisers.setup(opt, Flux.trainable(CGnet))

	num_epochs = 20
	batchsize = 256
	loss_history = Float64[]

	for epoch in 1:num_epochs
		epoch_loss = 0.0
		inds = shuffle(1:size(x_train_gpu, 1))

		for i in 1:batchsize:length(inds)
			idx = inds[i:min(i + batchsize - 1, end)]
			xb = x_train_gpu[idx, :]
			fb = f_train_gpu[idx]

			# Compute gradients and update weights
			grads = Zygote.gradient(Flux.params(CGnet)) do
				batch_loss_fast_gpu(xb, fb)
			end
			state = Optimisers.update!(state, Flux.trainable(CGnet), grads)

			# Accumulate loss for this batch
			epoch_loss += batch_loss_fast_gpu(xb, fb)
		end

		# Log total loss for this epoch
		push!(loss_history, epoch_loss)
	end
end

# ╔═╡ e873199b-575b-4f7d-90d2-32f60b847e7e
md"""
## Check Convergence Of Training
"""

# ╔═╡ ba8fce4e-3df8-4036-9ceb-d6021e01b8d8
plot(loss_history, xlabel="Epoch", ylabel="Loss", label="Training Loss", lw=2, title="CGnet Fast Training Loss")

# ╔═╡ d1c22034-ee3b-44e9-b9d9-a11797131867
md"""
## Visualize The Forces: True vs Learned
"""

# ╔═╡ 6796e99f-73ab-4f02-94dd-c64eb4284180
begin
xs_plot = collect(-5.5:0.1:5.5)
mean_force = [mean(fx_vals[i] for i in findall(abs.(x_vals .- x) .< 0.05)) for x in xs_plot]
pred_force = [-Zygote.gradient(x -> CGnet([x])[1], x)[1] for x in xs_plot]

plot(xs_plot, mean_force, label="Mean Force (Exact)", lw=2)
plot!(xs_plot, pred_force, label="CGnet Force", lw=2, ls=:dash)
scatter!(x_vals[1:1000], fx_vals[1:1000], label="Instantaneous Forces", alpha=0.3, ms=2)
xlabel!("x")
ylabel!("Force")
title!("Force Matching: CGnet vs Ground Truth")
end

# ╔═╡ b167a683-0a84-4196-bcb4-4a6913fc8435
md"""
## Visualize The Potential
"""

# ╔═╡ 535495c2-416d-40ec-a7c0-3242b7fdbe2a
begin
U_exact(x) = -log(sum(exp.(-[V(x, y) for y in -4:0.1:4])) + 1e-10)
U_net(x) = CGnet([x])[1]

U_exact_vals = [U_exact(x) for x in xs_plot]
U_net_vals = [U_net(x) for x in xs_plot]

plot(xs_plot, U_exact_vals .- minimum(U_exact_vals), label="Exact PMF", lw=2)
plot!(xs_plot, U_net_vals .- minimum(U_net_vals), label="CGnet PMF", lw=2, ls=:dash)
xlabel!("x")
ylabel!("Free Energy")
title!("Potential of Mean Force")
end

# ╔═╡ Cell order:
# ╟─1f58cc07-0388-4d62-a73d-5b352227111a
# ╟─d0fd2080-1985-11f0-00c2-bbfce2ad649b
# ╟─48bd4aa5-a6a5-4319-a6cb-821eebacc830
# ╠═d0c11460-6e24-4960-9e3c-c67c572272b4
# ╟─0fc0db9d-f5f6-439d-bfee-475e20d8a3ee
# ╠═7a63dbc2-c063-4866-a961-58ea3d594338
# ╟─6ed893e2-8186-4870-a22f-0da60934ff13
# ╠═0a67a981-8dbc-435f-a5b0-0a50485ea676
# ╟─ebe02d8f-b03a-4535-a695-7620e95ac598
# ╠═b5610bfd-c1b1-438c-bbad-074e45bc1578
# ╟─9e313963-c0ff-447e-86d1-9d96d503d5f5
# ╠═3746a4e2-62c3-4473-b7bc-eab0b07e42e5
# ╟─5cc05899-400f-4baf-a983-2d85cdac14d8
# ╠═5552f664-c1bb-4c83-89aa-dcea9a5fce12
# ╟─2880942e-288a-4fd3-9bc6-b20be2b9593d
# ╠═9d72129f-06ae-4b5f-9074-399094da26e5
# ╠═65a0e2c6-4e1c-4ed4-bec6-2c428d9b95df
# ╟─e873199b-575b-4f7d-90d2-32f60b847e7e
# ╠═ba8fce4e-3df8-4036-9ceb-d6021e01b8d8
# ╟─d1c22034-ee3b-44e9-b9d9-a11797131867
# ╠═6796e99f-73ab-4f02-94dd-c64eb4284180
# ╟─b167a683-0a84-4196-bcb4-4a6913fc8435
# ╠═535495c2-416d-40ec-a7c0-3242b7fdbe2a
