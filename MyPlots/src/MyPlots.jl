module MyPlots
using PyCall
using LaTeXStrings

export matplotlib, plt, plot, scatter

# @pyimport matplotlib.pyplot as plt
# plt=pyimport("matplotlib.pyplot")

const matplotlib = PyNULL()
const plt = PyNULL()

function __init__()
    copy!(matplotlib, pyimport("matplotlib"))
    copy!(plt, pyimport("matplotlib.pyplot")) # raw Python module
end

"""
# Example
    plot(x, y)
    plot(x, y, "b-")
    plot(sin, (0,π))
"""
function plot end

for plot_func in [:plot, :scatter]
    @eval begin
        function $plot_func(args...; kwargs...)
            plt.$plot_func(args...;  kwargs...)
        end

        function $plot_func(f::Function, x, args...; kwargs...)
            plt.$plot_func(x, f.(x), args...;  kwargs...)
        end
    end
end

function plot(f::Function, t::Tuple, args...; kwargs...)
    length(t) != 2 && error("Length of tuple != 2!")

    x = range(t..., length=1001)
    plt.plot(x, f.(x), args...;  kwargs...)
end

println("loaded MyPlots")
end
