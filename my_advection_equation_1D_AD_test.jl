using Test
using Enzyme
using ForwardDiff
using CircularArrays

function init1!(u::AbstractVector, x::AbstractVector)
	@. u[ x < -0.4] = 0.0
	@. u[-0.4 <= x < -0.2] = 1.0 - abs(x[-0.4 <= x < -0.2]+0.3) / 0.1
	@. u[-0.2 <= x < -0.1] = 0.0
	@. u[-0.1 <= x < -0.0] = 1.0
	@. u[ x >= 0.0 ] = 0.0
end


function upwind_non_circular(u::Vector)
    C = 1.1
    du = similar(u)
    for i in 2:length(u)
		du[i] =- 0.5C * (u[i]^2 -u[i-1]^2)  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
    du[1] = - 0.5C * (u[1]^2 -u[end]^2)  
    return du
end

function upwind_circular(u::CircularVector)
    C = 1.1
    up = similar(u)
	for i in eachindex(u)
		up[i] = u[i] - 0.5C * (u[i]^2 -u[i-1]^2)  # u_j^{n+1} = u_j^n - Δt/Δx * ( u_j^n - u_{j-1}^n )
	end
    du = up - u
end


x = range(-1.0, 2.0, step=0.01)
u=zeros(length(x))
c_u=CircularVector(0.0, length(x))

init1!(u, x)
init1!(c_u, x)

J1 = ForwardDiff.jacobian(upwind_non_circular, u)

# J2 = ForwardDiff.jacobian(upwind_circular, c_u)

J2 = Enzyme.jacobian(Forward, upwind_non_circular, u)
# J2 = Enzyme.jacobian(Forward, upwind_circular, c_u) # fail

@test J1 == J2 # Test Passed
