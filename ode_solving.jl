using Plots
using DifferentialEquations

function eulers_method(f::Function, α::Real, a::Real, b::Real, N::Int64)

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

function main()
    ode_solver_trial()
end

function ode_solver_trial()
    
    df(t,y) = y - t^2 + 1
    f(t) = -0.5*exp(t) + t^2 + 2t + 1

    u0 = 0.5
    a = 0
    b = 2
    n = 10
    tlin = 0:0.2:2

    euler_sol = eulers_method(df, u0, a, b, n)
    rk4_sol = runge_kutta_4(df, u0, a, b, n)

    # DE library
    df(u,p,t) = u - t^2 + 1
    tspan = (0., 2.)
    prob = ODEProblem(df, u0, tspan)
    de_lib_sol = solve(prob)

    plot(tlin, f.(tlin), label="exact", legend=:bottomright, dpi=150)
    plot!(de_lib_sol.t, de_lib_sol.u, markershape=:x, label="DE LIB")
    plot!(rk4_sol[:,1], rk4_sol[:,2], markershape=:+, label="rk4")
    plot!(euler_sol[:,1], euler_sol[:,1], markershape=:o, label="euler's")
end