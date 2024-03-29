using MyPlots


coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0  # maximum coordinate

initial_condition_sine_wave(x) = 1.0 + 0.5 * sin(π*x)



n_elements = 16 # number of elements
dx = (coordinates_max - coordinates_min) / n_elements # length of one element

plot(initial_condition_sine_wave, coordinates_min:dx:coordinates_max)

using Trixi
polydeg = 3 #= polynomial degree = N =#
basis = LobattoLegendreBasis(polydeg)

nodes = basis.nodes

weights = basis.weights

integral = sum(nodes.^3 .* weights)

x = Matrix{Float64}(undef, length(nodes), n_elements)
for element in 1:n_elements
    x_l = coordinates_min + (element - 1) * dx + dx/2
    for i in 1:length(nodes)
        ξ = nodes[i] # nodes in [-1, 1]
        x[i, element] = x_l + dx/2 * ξ
    end
end

u0 = initial_condition_sine_wave.(x)
using Plots
plot(vec(x), vec(u0), label="initial condition", legend=:topleft)

using LinearAlgebra
M = diagm(weights)
B = diagm([-1; zeros(polydeg - 1); 1])
D = basis.derivative_matrix
surface_flux = flux_lax_friedrichs

# %%
function rhs!(du, u, x, t)
    # Reset du and flux matrix
    du .= zero(eltype(du))
    flux_numerical = copy(du)

    # Calculate interface and boundary fluxes, $u^* = (u^*|_{-1}, 0, ..., 0, u^*|^1)^T$
    # Since we use the flux Lax-Friedrichs from Trixi.jl, we have to pass some extra arguments.
    # Trixi.jl needs the equation we are dealing with and an additional `1`, that indicates the
    # first coordinate direction.
    equations = LinearScalarAdvectionEquation1D(1.0)
    for element in 2:n_elements-1
        # left interface
        flux_numerical[1, element] = surface_flux(u[end, element-1], u[1, element], 1, equations)
        flux_numerical[end, element-1] = flux_numerical[1, element]
        # right interface
        flux_numerical[end, element] = surface_flux(u[end, element], u[1, element+1], 1, equations)
        flux_numerical[1, element+1] = flux_numerical[end, element]
    end
    # boundary flux
    flux_numerical[1, 1] = surface_flux(u[end, end], u[1, 1], 1, equations)
    flux_numerical[end, end] = flux_numerical[1, 1]

    # Calculate surface integrals, $- M^{-1} * B * u^*$
    for element in 1:n_elements
        du[:, element] -= (M \ B) * flux_numerical[:, element]
    end

    # Calculate volume integral, $+ M^{-1} * D^T * M * u$
    for element in 1:n_elements
        flux = u[:, element]
        du[:, element] += (M \ transpose(D)) * M * flux
    end

    # Apply Jacobian from mapping to reference element
    for element in 1:n_elements
        du[:, element] *= 2 / dx
    end

    return nothing
end

# %%

using Test
using Enzyme
using ForwardDiff

du = similar(u0)
rhs!(du, u0, x, 0.0)

function foo(u0)
    du = similar(u0)
    rhs!(du, u0, x, 0.0)
    du
end

J1 = ForwardDiff.jacobian(du, u0) do du_ode, u_ode
    rhs!(du_ode, u_ode, x, 0.0)
end
J2 = ForwardDiff.jacobian(foo, u0)

@test J1 == J2 # Test Passed

J = Enzyme.jacobian(Forward, foo, u0) # fail
