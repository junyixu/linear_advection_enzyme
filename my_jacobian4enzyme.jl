using Test
using Enzyme
using ForwardDiff
using CircularArrays

function my_jacobian_forward!(f!::Function, y::AbstractVector, x::AbstractVector)
    dx = Enzyme.onehot(x)
    dy = Tuple(zeros(size(x)) for _ in 1:length(x))
    Enzyme.autodiff(Forward, f!, BatchDuplicated(y, dy), BatchDuplicated(x, dx))
    return stack(dy)
end

function my_jacobian_reverse!(f!::Function, y::AbstractVector, x::AbstractVector)
    bx = Tuple(zeros(size(x)) for _ in 1:length(x))
    by = Enzyme.onehot(y)
    Enzyme.autodiff(Reverse, f!, BatchDuplicated(y, by), BatchDuplicated(x, bx))
    return stack(bx)'
end

# %%
# example 1

function f!(y::Vector{Float64}, x::Vector{Float64})
    y[1] = x[1] * x[1] + x[2] * x[1]
    y[2] = x[2] * x[2] + x[1] * x[2]
    return nothing
end

let 
    x  = [2.0, 3.0]
    y  = [0.0, 0.0]
    @test my_jacobian_reverse!(f!, y, x) == my_jacobian_forward!(f!, y, x) # Test Passed
end

# ∂y₁/∂x₁ ∂y₁/∂x₂ 
# ∂y₂/∂x₁ ∂y₂/∂x₂ 
#
# 2×2 Matrix{Float64}:
#  7.0  2.0
#  3.0  8.0

# %%
# example 2

function polar_coor!(xy::Vector{Float64}, rϕ::Vector{Float64})
    r = rϕ[1]
    ϕ = rϕ[2]
    xy[1] = r * cos(ϕ)
    xy[2] = r * sin(ϕ)
    return nothing
end

let
    rϕ = [2.0, π/2]
    xy = [0.0, 0.0]
    @test my_jacobian_reverse!(polar_coor!, xy, rϕ) == my_jacobian_forward!(polar_coor!, xy, rϕ) # Test Passed
end

# cosϕ -r*sinϕ
# sinϕ  r*cosϕ
#
# 2×2 Matrix{Float64}:
#  6.12323e-17  -2.0
#  1.0           1.22465e-16

# %%

