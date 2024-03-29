#=
	Created On  : 2023-04-03 00:24
    Copyright © 2023 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#

using CircularArrays
using PyCall
using LaTeXStrings

@pyimport matplotlib.pyplot as plt
@pyimport matplotlib
# https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
matplotlib.rc("font", size=14)

# %%
function init1(x::AbstractVector, u::AbstractVector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
end

function init2(x::AbstractVector, u::AbstractVector) # Burgers 方程 初始化
	@. u[ x < -0.8] = 1.8
	@. u[-0.8 <= x < -0.3] = 1.4 + 0.4*cos(2π*(x[-0.8 <= x < -0.3]+0.8))
	@. u[-0.3 <= x < 0.0] = 1.0
	@. u[ x >= 0.0 ] = 1.8
end

struct Cells
	x::AbstractVector{Float64}
	u::CircularVector{Float64} # u^n
	up::CircularVector{Float64} # u^(n+1) ; u plus
	function Cells(b::Float64=-1.0, e::Float64=2.0; step::Float64=0.01, init::Function=init1)
		x = range(b, e, step=step)
		u=CircularVector(0.0, length(x))
		init(x, u)
		up=similar(u)
		new(x, u , up)
	end
end
Cells(Δ::Float64)=Cells(-1.0, 2.0, step=Δ)
Cells(init::Function)=Cells(-1.0, 2.0, init=init)
Cells(b::Float64, e::Float64, Δ::Float64)=Cells(b, e, step=Δ)

next(c::Cells, flg::Bool)::CircularVector = flg ? c.up : c.u
current(c::Cells, flg::Bool)::CircularVector = flg ? c.u : c.up

function update!(c::Cells, flg::Bool, f::Function, C::AbstractFloat)
	up=next(c, flg) # u^(n+1)
	u=current(c, flg) # u^n
	f(up, u, C)
	return !flg
end
update!(c::Cells, flg::Bool, f::Function) = update!(c, flg, f, 0.05)

function minmod(a::AbstractFloat, b::AbstractFloat)::AbstractFloat
	if sign(a) * sign(b) > 0
		if abs(a) < abs(b)
			return a
		end
		return b
	end
	return 0
end


# %%
C = 0.95
Δx= 0.007
# C = Δt/Δx
Δt =  C * Δx


# function upwind(up::CircularVector, u::CircularVector, C::AbstractFloat)
# 	for i in eachindex(u)
# 		up[i] = u[i] - C * (u[i] -u[i-1])  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
# 	end
# end

function upwind(up::CircularVector, u::CircularVector, C::AbstractFloat)
	n = length(u)
	A=Matrix(Tridiagonal(repeat([-C], n-1), repeat([C+1], n), repeat([0.0], n-1)))
	A[1, n] = -C
	up .=  A \ u
end

function ygs(up::CircularVector, u::CircularVector, C::AbstractFloat)
	n = length(u)
	cc = zeros(n)
	for i in eachindex(cc)
		cc[i] = 0.5*C*(u[i] + u[i-1])
	end
	A=Matrix(Tridiagonal(-cc[2:end], cc.+1, repeat([0.0], n-1)))
	A[1, n] = -cc[1]
	up .=  A \ u
end



function upwind2(up::CircularVector, u::CircularVector, C::AbstractFloat)
	for i in eachindex(u)
		up[i] = u[i] - 0.5C * (u[i]^2 -u[i-1]^2)  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
end

function lax_wendroff(up::CircularVector, u::CircularVector, C::AbstractFloat)
	for j in eachindex(u)
		up[j] = u[j] - 0.5 * C * ( u[j+1] - u[j-1] ) + 0.5 * C^2 * ( u[j+1] - 2u[j] + u[j-1] )
	end
end

function limiter(up::CircularVector, u::CircularVector, C::AbstractFloat)
	for i in eachindex(u)
		up[i] = u[i] - C * (u[i] - u[i-1]) - 0.5 * C * (1 - C) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
end

function limiter2(up::CircularVector, u::CircularVector, C::AbstractFloat)
	for i in eachindex(u)
		B = 0.5C*(u[i] + u[i-1])
		up[i] = u[i] - B *  (u[i] - u[i-1]) - 0.5 * B * (1 - B) *
			( minmod(u[i]-u[i-1], u[i+1]-u[i]) - minmod(u[i-1]-u[i-2], u[i]-u[i-1]) )
	end
end



# %%



# %%
function problem1(C::AbstractFloat, f::Function, title::String; Δx::AbstractFloat=0.007)
	t=0.5
	Δt = Δx * C
	c=Cells(step=Δx, init=init1)
	plt.plot(c.x, c.u, "-.k", linewidth=0.2, label="init")
    plt.plot(c.x, circshift(c.u, round(Int, t*C/Δt)), "-g", linewidth=1, alpha=0.4)

	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update!(c, flg, f, C)
	end

	plt.title("time = "*string(t)*", "*"C = "*string(C)*", "* title)
	plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.savefig("../figures/problem1_"*string(f)*string(C)*".pdf", bbox_inches="tight")
	plt.show()
end

# %%
function problem2(t::AbstractFloat)
	C = 1.1
	Δt =  C * Δx
	f = ygs
	matplotlib.rc("font", size=13)
	plt.figure(figsize=(10,2.5))
	c=Cells(step=Δx, init=init2)
	plt.plot(c.x, c.u, "-.k", linewidth=0.2, label="init")
    # plt.plot(c.x, circshift(c.u, round(Int, t*C/Δt)), "-g", linewidth=1, alpha=0.4)
	flg=true # flag
	for _ = 1:round(Int, t/Δt)
		flg=update!(c, flg, f, C)
	end
	plt.title("time = "*string(t)*", "*"Minmod")
	# plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, marker="o", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	plt.plot(c.x, c.up, linestyle="dashed", linewidth=0.4, color="navy", marker="o", markeredgecolor="purple", markeredgewidth=0.4, markersize=4,  markerfacecolor="none", label="up")
	# plt.savefig("../figures/problem2_"*string(f)*string(t)*".pdf", bbox_inches="tight")
	plt.show()
end

# %%
function main()
	problem1(0.05, upwind, "Upwind")
	problem1(1.0, upwind, "Upwind")
	problem1(0.01, upwind, "Upwind")
	problem1(1.0, upwind, "Upwind")
	problem1(0.95, lax_wendroff, "Lax-Wendroff")
	problem1(0.95, limiter, "Minmod")
	problem2(0.25)
	problem2(0.5)
	problem2(0.75)
	problem2(1.0)
end
main()
