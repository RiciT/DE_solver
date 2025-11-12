using Plots
using DifferentialEquations

function EulersMethod(f::Function, α::Real, a::Real, b::Real, N::Int64)

    #number of steps
    n1 = N + 1
    #first column is the time-series and second is the solutions
    u = zeros(n1, 2)

    #step-size
    h = (b - a) / N
    #initial conditions
    u[1,1] = a
    u[1,2] = α

    #actual Euler's method implementation
    for i in 2:n1
        u[i,2] = u[i - 1, 2] + h * f(u[i - 1, 1], u[i - 1, 2])
        u[i,1] = a + (i - 1) * h
    end

    return u
end

function runge_kutta_4(f::Function, α::Real, a::Real, b::Real, N::Int64)

    #number of steps
    n1 = N + 1
    #first column is time-series and second is solution steps
    u = zeros(n1, 2)

    #step-size
    h = (b - a) / N
    #inital conditions
    u[1,1] = a
    u[1,2] = α

    #RK4 method
    for i in 2:n1
        t = u[i - 1,1]
        w = u[i - 2,2]

        k1 = h * f(t, w)
        k2 = h * f(t + h / 2, w + k1 / 2)
        k3 = h * f(t + h / 2, w + k2 / 2)
        k4 = h * f(t + h, w + k3)

        u[i,2] = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u[i,1] = a + (i - 1) * h
    end

    return u
end