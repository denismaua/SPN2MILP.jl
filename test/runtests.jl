# Test SPN to MILP translation
using Test
# using BenchmarkTools

# ENV["GUROBI_HOME"] = "/Library/gurobi902/mac64"

using SPN2MILP

@testset "SPN2MILP" begin
    using SumProductNetworks
    import SumProductNetworks: SumNode, ProductNode, CategoricalDistribution, IndicatorFunction, vardims
    using Gurobi

    # SPN with dichotomic sum nodes (<= 2 children)
    SPN = SumProductNetwork(
        [
            SumNode([3,2],[0.2,0.8]),               # 1
            SumNode([4,5],[0.625,0.375]),           # 2
            ProductNode([6,8]),                     # 3
            ProductNode([6,9]),                     # 4
            ProductNode([7,9]),                     # 5
            SumNode([10,11],[0.6,0.4]),             # 6
            SumNode([10,11],[0.1,0.9]),             # 7
            SumNode([12,13],[0.3,0.7]),             # 8
            SumNode([12,13],[0.8,0.2]),             # 9
            IndicatorFunction(1,1),                 # 10
            IndicatorFunction(1,2),                 # 11
            IndicatorFunction(2,1),                 # 12
            IndicatorFunction(2,2),                 # 13
        ]
    )
    params = Dict{String,Any}(
        "IterationLimit" => 100, # Simplex iteration limit
        "Method" => 2, # Algorithm used to solve continuous models (default: -1 -> automatic, 
                       #                                                0 -> primal simplex, 
                       #                                                1 -> dual simplex, 
                       #                                                2 -> barrier, 
                       #                                                3 -> concurrent, 
                       #                                                4 -> deterministic concurrent, 
                       #                                                5 -> deterministic concurrent simplex
        "TimeLimit" => 100,
        "IntFeasTol" => 1e-9, # Integer feasibility tolerance (default: 1e-5, minimum: 1e-9, maxL 1e-1)
        "BarConvTol" => 1e-8, # Barrier convergence tolerance (Default: 1e-8, min: 0, max: 1)
        "OptimalityTol" => 1e-6, # Dual feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "FeasibilityTol" => 1e-9, # Primal feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "MIPGap" => 0, # Relative optimality gap (default: 1e-4, min: 0, max: Inf)
        "MIPGapAbs" => 0, # Absolute MIP optimality gap (default: 1e-10, min: 0, max: Inf)
        "Heuristics" => 0.1, # Time spent in feasibility heurstics (default: 0.05, min: 0, max: 1)
        "MIPFocus" => 0, # MIP solver focus (default: 0, 1 -> find feasible solutions, 2 -> focus proving optimlaity, 3 -> focus on improving bound)
        "Presolve" => -1, # Controls the presolve level (default: -1 -> automatic, 0 -> off, 1 -> conservative, 2 -> aggressive)
        "FeasRelaxBigM" => 1e6, # Big-M value for feasibility relaxations (default: 1e6, min:0, max: Inf)
        "Quad" => -1, # Controls quad precision in simplex (default: -1 -> automatic, 0 -> off, 1 -> on)
        )
    model = spn2milp(SPN, [6,7,2,9,1,8], params)
    # println(model)
    optimize(model)
    # get status
    st = get_status(model)
    println("Status: $(st)")
    # show results
    sol = get_solution(model)
    println("Solution: $(sol)")
    # parse solution to extract MAP assignment
    vdims = vardims(SPN)
    x = Array{Float64}(undef, length(vdims))
    fill!(x, NaN)
    offset = 0
    for var in sort(collect(keys(vdims)))
        for value = 1:vdims[var]
            offset += 1
            if sol[offset] ≈ 1.0
                x[var] = value
            end
        end
    end
    # println(x)
    @test (x[1] == 2.0) && (x[2] == 1.0)
    println("Solution value: ", SPN(x))
    @test SPN(x) ≈ 0.4
    # show obj value
    obj = get_objval(model)
    println("Objective: ", obj)
    # show obj bound
    bound = Gurobi.get_objbound(model)
    println("Upper bound: ", bound)
    @test obj ≈ 0.4
end