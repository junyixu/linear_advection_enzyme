#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    test
    Copyright © 2024 Junyi Xu <jyxu@mail.ustc.edu.cn>

    Distributed under terms of the MIT license.
=#


using Trixi, LinearAlgebra, Plots

equations = CompressibleEulerEquations2D(1.4)
solver = DGSEM(3, flux_central)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver)
J = jacobian_ad_forward(semi);
JJ=Trixi.jacobian_ad_forward_enzyme(semi)
size(J)

t0=zero(real(semi))
u0_ode = compute_coefficients(t0, semi)
du_ode = similar(u0_ode)
Trixi.rhs!(du_ode, u0_ode, semi, t0)
du_ode

t0=zero(real(semi))
u0_ode = compute_coefficients(t0, semi)

function f(u0_ode)
    du_ode = similar(u0_ode)
    Trixi.rhs!(du_ode, u0_ode, new_semi, t0)
    du_ode
end

using ForwardDiff
config = ForwardDiff.JacobianConfig(nothing, du_ode, u0_ode)


new_semi = Trixi.remake(semi, uEltype = eltype(config))

J = ForwardDiff.jacobian(du_ode, u0_ode, config) do du_ode, u_ode
    Trixi.rhs!(du_ode, u_ode, new_semi, t0)
end

using ForwardDiff
using Enzyme
J = ForwardDiff.jacobian(du_ode, u0_ode, config) do du_ode, u_ode
    Trixi.rhs!(du_ode, u_ode, new_semi, t0)
end

t0=zero(real(semi))
u0_ode = compute_coefficients(t0, semi)
config = ForwardDiff.JacobianConfig(nothing, u0_ode)
new_semi = Trixi.remake(semi, uEltype = eltype(config))
function foo(u0_ode)
    du_ode = similar(u0_ode)
    Trixi.rhs!(du_ode, u0_ode, new_semi, t0)
    du_ode 
end
J = ForwardDiff.jacobian(foo, u0_ode, config) # success

using Enzyme
J = jacobian(Forward, foo, u0_ode) # fail
