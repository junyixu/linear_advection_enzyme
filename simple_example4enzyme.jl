function g(x::AbstractVector)
    return log(x[1]) + x[1]*x[2] - sin(x[2])
end

x = [2.0, 5.0]
∂g_∂x= zeros(size(x))
autodiff(Reverse, g, Active, Duplicated(x, ∂g_∂x))

# julia> ∂g_∂x
# 2-element Vector{Float64}:
#  5.5
#  1.7163378145367738

function g!(rst_vec::AbstractVector, x::AbstractVector)
    rst_vec[1] = log(x[1]) + x[1]*x[2] - sin(x[2])
    return nothing
end
x = [2.0, 5.0]
dx = [0.0, 0.0]
y = [0.0]
dy = [1.0]
autodiff(Reverse, g!, Const, Duplicated(y, dy), Duplicated(x, dx))

# julia> dx
# 2-element Vector{Float64}:
#  5.5
#  1.7163378145367738
