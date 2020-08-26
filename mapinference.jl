using SPN2MILP
using SumProductNetworks
import SumProductNetworks: vardims
using Gurobi

function run(spn_filename,q_filename, loadfromfile=false)
    maxinstances = 10
    println("SPN: ", spn_filename)
    println("Query: ", q_filename)
    println()
    # Load SPN form file
    spn = SumProductNetwork(spn_filename; offset = 1)
    println(summary(spn))
    # linear ordering of values of variables (≈ indicator nodes)
    vdims = vardims(spn)
    offset = 0
    optvar = Dict{Tuple{Int,Int},Int}()
    for var in sort(collect(keys(vdims)))
        for value = 1:vdims[var]
            offset += 1
            optvar[(var,value)] = offset
        end
    end
    # println(optvar)
    @assert length(optvar) == offset
    # to store solution
    x = Array{Float64}(undef, length(vdims))
    fill!(x, NaN)
    # Gurobi parameters 
    params = Dict{String,Any}(
        # "IterationLimit" => 100, # Simplex iteration limit
        "Method" => 1, # Algorithm used to solve continuous models (default: -1 -> automatic, 
                       #                                                0 -> primal simplex, 
                       #                                                1 -> dual simplex, 
                       #                                                2 -> barrier, 
                       #                                                3 -> concurrent, 
                       #                                                4 -> deterministic concurrent, 
                       #                                                5 -> deterministic concurrent simplex
        # "TimeLimit" => 100,
        "IntFeasTol" => 1e-9, # Integer feasibility tolerance (default: 1e-5, minimum: 1e-9, max: 1e-1)
        "BarConvTol" => 1e-8, # Barrier convergence tolerance (Default: 1e-8, min: 0, max: 1)
        "OptimalityTol" => 1e-6, # Dual feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "FeasibilityTol" => 1e-9, # Primal feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "MIPGap" => 0, # Relative optimality gap (default: 1e-4, min: 0, max: Inf)
        "MIPGapAbs" => 0, # Absolute MIP optimality gap (default: 1e-10, min: 0, max: Inf)
        # "Heuristics" => 0.1, # Time spent in feasibility heuristics (default: 0.05, min: 0, max: 1)
        "MIPFocus" => 3, # MIP solver focus (default: 0 -> balanced, 1 -> find feasible solutions, 2 -> focus proving optimality, 3 -> focus on improving bound)
        "Presolve" => 0, # Controls the presolve level (default: -1 -> automatic, 0 -> off, 1 -> conservative, 2 -> aggressive)
        # "FeasRelaxBigM" => 1e6, # Big-M value for feasibility relaxations (default: 1e6, min:0, max: Inf)
        "Quad" => 1, # Controls quad precision in simplex (default: -1 -> automatic, 0 -> off, 1 -> on)
        )
    if loadfromfile && isfile(spn_filename * ".mps")
        # read it from file if it exists
        env = Gurobi.Env()
        for (param,value) in params
            Gurobi.setparam!(env, param, value)
        end
        model = Gurobi.Model(env, "milp", :maximize)
        # timetaken = @elapsed Gurobi.read_model(model, spn_filename * ".lp")
        timetaken = @elapsed Gurobi.read_model(model, spn_filename * ".mps")
    else
        # else, run variable elimination to generate milp model
        timetaken = @elapsed model = spn2milp(spn, nothing, params)
        println("MILP model build in $(timetaken)s.")
        # Writing to file
        Gurobi.write_model(model, spn_filename * ".lp")
        Gurobi.write_model(model, spn_filename * ".mps")
    end
    # Load query, evidence and marginalized variables
    totaltime = @elapsed open(q_filename) do io
        inst = 1
        # printstyled("╮\n"; color = :red)
        while !eof(io)
            basemodel = Gurobi.copy(model)
            # fill!(x, NaN) # reset configuration
            printstyled("(", inst, ")\n"; color = :red)
            # Read query variables
            fields = split(readline(io))
            header = fields[1]
            @assert header == "q"
            query = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
            for var in query
                # add_constr!(model, [ optvar[(var,value)] for value=1:vdims[var] ], ones(Float64,vdims[var]), '=', 1.0)
                # exactly one value must be selected:
                Gurobi.add_sos!(model, :SOS1, [ optvar[(var,value)] for value=1:vdims[var] ], ones(Float64,vdims[var]))
            end
            # Read marginalized variables
            fields = split(readline(io))
            header = fields[1]
            @assert header == "m"
            marg = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
            for var in marg
                x[var] = NaN
                # println("marg ", var)
                for j = 1:vdims[var]
                    add_constr!(model, [ optvar[(var,j)] ], [1.0], '=', 1.0)
                end
            end
            # Read evidence
            fields = split(readline(io))
            header = fields[1]
            @assert header == "e"
            for i=2:2:length(fields)
                var = parse(Int,fields[i]) + 1
                value = parse(Int,fields[i+1]) + 1
                x[var] = value
                # println(var, '=', value)
                for j = 1:vdims[var]
                    if j == value
                        add_constr!(model, [ optvar[(var,value)] ], [1.0], '=', 1.0)
                    else
                        add_constr!(model, [ optvar[(var,j)] ], [1.0], '=', 0.0)
                    end
                end
                # println('x',optvar[(var,value)], '=', 1.0)
            end
            # TODO: add MIP start from maxproduct solution or other
            # Gurobi.set_dblattrelement!(model, "Start", idx, value)
            # Alternatively, we can read a MIP start from file (MST) with Gurobi.read
            update_model!(model)
            # model2 = Gurobi.presolve_model(model)
            optimize(model)
            # get status
            st = get_status(model)
            println("Status: $(st)")
            if st == :inf_or_unbd || st == :unbounded || st == :infeasible
                @warn "Infeasible or unbounded program; could not extract a solution."
            else                
                # parse solution to extract MAP assignment
                sol = get_solution(model)
                for var in query
                    for value = 1:vdims[var]
                        if sol[optvar[(var,value)]] ≈ 1.0
                            x[var] = value
                            break
                        end
                    end
                end
                # for (k,v) in optvar
                #     if sol[v] ≈ 1.0
                #         x[k[1]] = k[2]
                #     end
                # end
                println("Solution value: ", spn(x))
                # show obj value
                obj = get_objval(model)
                println("Objective: ", obj)
                # show obj bound
                bound = Gurobi.get_objbound(model)
                println("Upper bound: ", bound)
            end
            inst += 1
            # if maximum no. of instances is reached, stop
            if inst > maxinstances
                break
            end
            Gurobi.free_model(model)
            model = basemodel
        end
    end
    nothing
end

run("/Users/denis/code/SPN/spambase.spn2", "/Users/denis/code/SPN/spambase.map")
# run("/Users/denis/code/example.spn", "/Users/denis/code/example.map", true)