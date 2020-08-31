using SPN2MILP
using SumProductNetworks
import SumProductNetworks: vardims, project, project2
import SumProductNetworks.MAP: maxproduct!
using Gurobi

"Exact inference"
function run(spn_filename, q_filename, multiplier=1.0, maxinstances=1, verbose=true, loadfromfile=false)
    println("SPN: ", spn_filename)
    println("Query: ", q_filename)
    println("Multiplier: ", multiplier)
    println("Verbose: ", verbose)
    println("Maximum instances: ", maxinstances)
    println()
    # Load SPN form file
    spn = SumProductNetwork(spn_filename; offset = 1)
    summary(spn)
    println()
    # linear ordering of values of variables (≈ indicator nodes)
    # offset = 0
    # optvar = Dict{Tuple{Int,Int},Int}()
    # for var in sort(collect(keys(vdims)))
    #     for value = 1:vdims[var]
    #         offset += 1
    #         optvar[(var,value)] = offset
    #     end
    # end
    # println(optvar)
    # @assert length(optvar) == offset
    # to store solution
    x = Array{Float64}(undef, length(scope(spn)))
    fill!(x, NaN)
    # Gurobi parameters 
    params = Dict{String,Any}(
        # "IterationLimit" => 100, # Simplex iteration limit
        "Method" => -1, # Algorithm used to solve continuous models (default: -1 -> automatic, 
                       #                                                0 -> primal simplex, 
                       #                                                1 -> dual simplex, 
                       #                                                2 -> barrier, 
                       #                                                3 -> concurrent, 
                       #                                                4 -> deterministic concurrent, 
                       #                                                5 -> deterministic concurrent simplex
        "TimeLimit" => 3600,
        "NumericFocus" => 3, # Numerical precision (default: 0 -> automatic, 1-3 increase accuracy)
        "IntFeasTol" => 1e-9, # Integer feasibility tolerance (default: 1e-5, minimum: 1e-9, max: 1e-1)
        "BarConvTol" => 1e-22, # Barrier convergence tolerance (Default: 1e-8, min: 0, max: 1)
        "OptimalityTol" => 1e-9, # Dual feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "FeasibilityTol" => 1e-9, # Primal feasibility tolerance (default: 1e-6, min: 1e-9, max: 1e-2)
        "MIPGap" => 0, # Relative optimality gap (default: 1e-4, min: 0, max: Inf)
        "MIPGapAbs" => 0, # Absolute MIP optimality gap (default: 1e-10, min: 0, max: Inf)
        # "Heuristics" => 0.1, # Time spent in feasibility heuristics (default: 0.05, min: 0, max: 1)
        "MIPFocus" => 0, # MIP solver focus (default: 0 -> balanced, 1 -> find feasible solutions, 2 -> focus proving optimality, 3 -> focus on improving bound)
        "Presolve" => 2, # Controls the presolve level (default: -1 -> automatic, 0 -> off, 1 -> conservative, 2 -> aggressive)
        "FeasRelaxBigM" => 1e20, # Big-M value for feasibility relaxations (default: 1e6, min:0, max: Inf)
        "Quad" => -1, # Controls quad precision in simplex (default: -1 -> automatic, 0 -> off, 1 -> on)
        )
    # if loadfromfile && isfile(spn_filename * ".mps")
    #     # read it from file if it exists
    #     env = Gurobi.Env()
    #     for (param,value) in params
    #         Gurobi.setparam!(env, param, value)
    #     end
    #     model = Gurobi.Model(env, "milp", :maximize)
    #     # timetaken = @elapsed Gurobi.read_model(model, spn_filename * ".lp")
    #     timetaken = @elapsed Gurobi.read_model(model, spn_filename * ".mps")
    # else
    #     # else, run variable elimination to generate milp model
        # timetaken = @elapsed model = spn2milp(spn, :deg, params, 1.)
        # println("MILP model build in $(timetaken)s.")
    #     # Writing to file
    #     Gurobi.write_model(model, spn_filename * ".lp")
    #     Gurobi.write_model(model, spn_filename * ".mps")
    # end
    # Load query, evidence and marginalized variables
    totaltime = @elapsed open(q_filename) do io
        inst = 1
        # printstyled("╮\n"; color = :red)
        while !eof(io)
            # basemodel = Gurobi.copy(model)
            fill!(x, NaN) # reset configuration
            printstyled("(", inst, ")\n"; color = :red)
            # Read query variables
            fields = split(readline(io))
            header = fields[1]
            @assert header == "q"
            query = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
            # Read marginalized variables
            fields = split(readline(io))
            header = fields[1]
            @assert header == "m"
            marg = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
            # Read evidence
            fields = split(readline(io))
            header = fields[1]
            @assert header == "e"
            evidence = Dict{UInt, Float64}()
            for i=2:2:length(fields)
                var = parse(Int,fields[i]) + 1
                value = parse(Int,fields[i+1]) + 1
                x[var] = value
                evidence[var] = value
                # println(var, '=', value)
            end
            # Build specialized model (remove marginalized variables)
            # println("Query: ", query)
            # println("Marginalized: ", marg)
            # println("Evidence:", evidence)
            # spn2 = project(spn, union(query, keys(evidence)), x)
            spn2 = project2(spn, query, x)
            println(summary(spn2))
            timetaken = @elapsed model = SPN2MILP.spn2milp_q(spn2, query, evidence, :deg, params, multiplier, verbose)
            println("MILP model build in $(timetaken)s.")
            # Write model to file
            Gurobi.write_model(model, "$(spn_filename)-$(inst).mps")
            # for var in query
            #     # exactly one value must be selected:
            #     Gurobi.add_sos!(model, :SOS1, [ optvar[(var,value)] for value=1:vdims[var] ], ones(Float64,vdims[var]))
            # end            
            # for var in marg
            #     x[var] = NaN
            #     # println("marg ", var)
            #     for j = 1:vdims[var]
            #         add_constr!(model, [ optvar[(var,j)] ], [1.0], '=', 1.0)
            #     end
            # end   
            # for (var,value) in evidence
            #     for j = 1:vdims[var]
            #         if j == value
            #             add_constr!(model, [ optvar[(var,value)] ], [1.0], '=', 1.0)
            #         else
            #             add_constr!(model, [ optvar[(var,j)] ], [1.0], '=', 0.0)
            #         end
            #     end
            #     # println('x',optvar[(var,value)], '=', 1.0)
            # end
            # Run maxproduct to obtain initial solution
            maxproduct!(x, spn2, query);
            printstyled("MaxProduct: "; color = :green)
            println(spn(x))
            vdims = vardims(spn2)
            optvar = Dict{Tuple{Int,Int},Int}()
            offset = 0
            sc = scope(spn2)
            for var in sort!(collect(query) ∩ sc)
                for value = 1:vdims[var]
                    offset += 1
                    optvar[(var,value)] = offset
                end
            end 
            for var in sort!(collect(keys(evidence)) ∩ sc)
                for value = 1:vdims[var]
                    offset += 1
                    optvar[(var,value)] = offset
                end
            end             
            # Set MIP Start solution
            for var in query ∩ sc
                Gurobi.set_dblattrelement!(model, "Start", optvar[(var,x[var])], 1.0)
                # for value = 1:vdims[var]
                #     if x[var] == value
                #         Gurobi.set_dblattrelement!(model, "Start", optvar[(var,value)], 1.0)
                #     else
                #         Gurobi.set_dblattrelement!(model, "Start", optvar[(var,value)], 0.0)
                #     end
                # end
            end
            # Alternatively, we can read a MIP start from file (MST) with Gurobi.read
            update_model!(model)
            # model2 = Gurobi.presolve_model(model)
            optimize(model)
            # get status
            st = get_status(model)
            println("Status: $(st)")
            Gurobi.write_model(model, "$(spn_filename)-$(inst).json")
            if st == :inf_or_unbd || st == :unbounded || st == :infeasible
                @warn "Infeasible or unbounded program; could not extract a solution."
            else                                
                # parse solution to extract MAP assignment                              
                sol = get_solution(model)
                # Write to file
                # Gurobi.write_model(model, "$(spn_filename)-$(inst).sol")
                Gurobi.write_model(model, "$(spn_filename)-$(inst).mst")
                for var in query ∩ sc
                    for value = 1:vdims[var]
                        if sol[optvar[(var,value)]] ≈ 1.0
                            x[var] = value
                            # break
                        end
                    end
                end
                printstyled("Solution value: "; color = :light_cyan)
                println(spn(x))
                # show obj value
                obj = get_objval(model)
                printstyled("Objective: "; color = :light_cyan)
                println(obj)
                # show obj bound
                bound = Gurobi.get_objbound(model)
                printstyled("Upper bound: "; color = :light_cyan)
                println(bound)
            end
            inst += 1
            # if maximum no. of instances is reached, stop
            if inst > maxinstances
                break
            end
            Gurobi.free_model(model)
            # model = basemodel
        end
    end
    totaltime
end

if length(ARGS) == 2
    @time run(ARGS[1], ARGS[2])
elseif length(ARGS) == 3
    @time run(ARGS[1], ARGS[2], parse(Float64, ARGS[3]))
elseif length(ARGS) == 4
    @time run(ARGS[1], ARGS[2], parse(Float64, ARGS[3]), parse(Int, ARGS[4])) 
elseif length(ARGS) == 5
    @time run(ARGS[1], ARGS[2], parse(Float64, ARGS[3]), parse(Int, ARGS[4]), parse(Bool, ARGS[5])) 
else
    println("Insufficient arguments.")
end
# @time run("/Users/denis/code/SPN/spambase.spn2", "/Users/denis/code/SPN/spambase.map", 100., 1, true)
# run("/Users/denis/code/SPN/mushrooms.spn2", "/Users/denis/code/SPN/mushrooms_scenarios.map")
# run("/Users/denis/code/SPN/dna.spn2", "/Users/denis/code/SPN/dna.map")
# @time run("/Users/denis/code/SPN/nltcs.spn2", "/Users/denis/code/SPN/nltcs_scenarios.map", 100000., 1, true)
# @time run("/Users/denis/code/SPN/molecular-biology_promoters.spn2", "/Users/denis/code/SPN/molecular-biology_promoters_scenarios.map", 1e20, 1, true)
# run("/Users/denis/code/example.spn", "/Users/denis/code/example.map")