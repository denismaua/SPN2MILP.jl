module SPN2MILP

# Implementation of SPN2MILP algorithm
import AlgebraicDecisionDiagrams
# Aliases
const ADD = AlgebraicDecisionDiagrams
const MLExpr = ADD.MultilinearExpression
import Gurobi

"""
    spn2milp(spn::SumProductNetwork)

Translates sum-product network `spn` into MAP-equivalent mixed-integer linear program.
Require that sum nodes have exactly two children.
"""
function spn2milp(spn::SumProductNetwork, ordering::Union{Nothing,Array{<:Integer}}=nothing)    
    # obtain scope of every node
    scopes = SumProductNetworks.scopes(spn)
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node / latent variable
    buckets = Dict{Int,Array{ADD.DecisionDiagram{MLExpr}}}( i => [] for i in sumnodes ) 
    # TODO: Apply min-fill or min-degree heuristic to obtain better elimination ordering
    # variable elimination sequence
    if isnothing(ordering) # if no ordering is given, obtain one
        ordering = sort(sumnodes, rev=true) # eliminate bottom-most variables first
    end
    @assert length(ordering) == length(sumnodes) 
    vorder = Dict{Int,Int}() # ordering of elimination of each variable (inverse mapping)
    for i=1:length(ordering)
        vorder[ordering[i]] = i
    end
    # Create optimization model (interacts with Gurobi.jl)
    env = Gurobi.Env()
    # TODO: allow passing of parameters to solve
    # setparam!(env, "Method", 2)   # choose to use Barrier method
    # setparams!(env; IterationLimit=100, Method=1) # set the maximum iterations and choose to use Simplex method
     # creates an empty model ("milp" is the model name)
    model = Gurobi.Model(env, "milp", :maximize)

    ## First obtain ADDs for manifest variables
    offset = 0 # offset to apply to variable indices at ADD leaves
    vdims = SumProductNetworks.vardims(spn) # var id => no. of values
    for var in sort(scopes[1])
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        # Create corresponding optimization variables for leaves (interacts with Gurobi)
        # TODO: map optimization variables to spn variable assignment (var,value)
        # map(t -> begin
        #     # Gurobi.add_bvar!(model, 0.0)
        #     println("binary ", ADD.value(t))
        #     end, 
        #         Base.filter(n -> isa(n,ADD.Terminal), collect(α))
        #     )              
        for i=1:vdims[var]
            # syntax: model, coefficient in objective
            Gurobi.add_bvar!(model, 0.0)
        end
        # add constraint (interacts with Gurobi)
        # Gurobi.add_constr!(model, collect((offset+1):vdims[var]), '=', 1.0)
        idx = collect((offset+1):(offset+vdims[var]))
        coeff = ones(Float64, length(idx))
        # print(idx, coeff)
        # syntax: variables ids, coefficients, comparison (<,>,=), right-hand side constant
        Gurobi.add_constr!(model, idx, coeff, '=', 1.0)
        #
        offset += vdims[var] # update start index for next variable
        # # get index of bottom-most variable (highest id of a sum node)
        # id = maximum(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        # # get index of lowest variable according to elimination ordering
        i,id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        @assert length(buckets[id]) == 0 # TODO iterate until finding an empty bucket      
        # associate ADD to corresponding bucket
        push!(buckets[id],α)
        end
    ## Then build ADDs for sum nodes (latent variables)
    for id in sumnodes
        # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[id],α)
    end
    
    # To map each expression in a leaf into a fresh monomial
    cache = Dict{MLExpr,MLExpr}()
    bilinterms = Dict{ADD.Monomial,Int}()
    function renameleaves(e::MLExpr) 
        # get!(cache,e,MLExpr(1.0,offset+length(cache)+1))
        # If cached, do nothing
        if haskey(cache, e)
            return cache[e]
        end
        # Generate corresponding variable and constraint (interacts with Gurobi)
        f = MLExpr(1.0,offset+1)  
        # syntax is model, coeefficient in objective, [lowerbound, upper bound]
        Gurobi.add_cvar!(model, 0.0) # is it worth adding lower bounds? upper bounds?
        offset += 1 # increase opt var counter
        # println("continuous ", f)
        idx = [offset] # indices of variables in constraint
        coeff = [-1.0] # coefficients in linear constraint
        for (m,c) in e
            if length(m.vars) == 1
                push!(idx, m.vars[1])
            else # w = x*y
                @assert length(m.vars) == 2
                # Assumes both variables have domain [0,1]
                # Might lead to unfeasible program if this is violated
                if haskey(bilinterms,m)
                    id = bilinterms[m] # id of w
                else # add continuous variable w to problem with 0 ≤ w ≤ 1
                    Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # add new variable representing bilinear term
                    offset += 1
                    id = offset # id of w
                    bilinterms[m] = id
                end
                push!(idx, id)
                # w - x ≤ 0 
                Gurobi.add_constr!(model,[id, m.vars[1]], [1.0, -1.0], '<', 0.0)
                # w - y ≤ 0 
                Gurobi.add_constr!(model,[id, m.vars[2]], [1.0, -1.0], '<', 0.0)
                # w - y - x ≥ -1
                Gurobi.add_constr!(model,[id, m.vars[1], m.vars[2]], [1.0, -1.0, -1.0], '>', -1.0)
            end
            push!(coeff, c)
        end
        # println(idx, coeff)
        Gurobi.add_constr!(model, idx, coeff, '=', 0.0)
        # println("$f = $e")
        cache[e] = f
    end
    # Run variable elimination to generate constraints
    for i = 1:(length(ordering)-1)
        var = ordering[i] # variable to eliminate
        printstyled("Eliminate: ", var, "\n"; color = :red)
        α = reduce(*, buckets[var]; init = ADD.Terminal(MLExpr(1.0)))
        α = ADD.marginalize(α, var)        
        # Obtain copy with modified leaves and generate constraints (interacts with JUMP / Gurobi)
        β = ADD.apply(renameleaves, α)
        # For path decomposition, add "message" to next bucket to be processed
        # printstyled("-> $(ordering[i+1])\n"; color = :green)     
        push!(buckets[ordering[i+1]], β)
        # Create corresponding optimization variables for leaves (interacts with JUMP / Gurobi)
        # map(t -> begin
        # JuMP.@variable(model, base_name="$t", binary=true)
        # end, 
        #     Base.filter(n -> isa(n,ADD.Terminal), collect(β))
        # )  
        # println(β)
        # Print out constraint
        # TODO: replace by symbolic constraint representation (interact with JUMP / Gurobi)
        # ADD.apply(genconstraint, β, α)
        # For standard bucket elimination (generates tree-decomposition)
        # scope = Base.filter(n -> isa(n,ADD.Node), collect(α))
        # if isempty(scope)
        #     id = ordering[end]
        # else
        #     _, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), scope)  
        # end   
        # push!(buckets[id], β) 
        # printstyled("-> $id\n"; color = :green)     
        # 
        # for α in buckets[var]
        #     println(α)
        # end
    end
    # Objective (last variable elimination)
    α = reduce(*, buckets[ordering[end]]; init = ADD.Terminal(MLExpr(1.0)))
    α = ADD.marginalize(α, ordering[end])
    @assert isa(α,ADD.Terminal)
    # Add equality constraint to represent objective
    Gurobi.add_cvar!(model, 1.0)
    offset += 1
    idx = [offset]
    coeff = [-1.0]
    for (m, c) in ADD.value(α)
        if length(m.vars) == 1
            push!(idx, m.vars[1])
        else # w = x*y
            @assert length(m.vars) == 2
            # Assumes both variables have domain [0,1]
            # Might lead to unfeasible program if this is violated
            if haskey(bilinterms,m)
                id = bilinterms[m] # id of w
            else # add continuous variable w to problem with 0 ≤ w ≤ 1
                Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # add new variable representing bilinear term
                offset += 1
                id = offset # id of w
                bilinterms[m] = id
            end
            push!(idx, id)
            # w - x ≤ 0 
            Gurobi.add_constr!(model,[id, m.vars[1]], [1.0, -1.0], '<', 0.0)
            # w - y ≤ 0 
            Gurobi.add_constr!(model,[id, m.vars[2]], [1.0, -1.0], '<', 0.0)
            # w - y - x ≥ -1
            Gurobi.add_constr!(model,[id, m.vars[1], m.vars[2]], [1.0, -1.0, -1.0], '>', -1.0)
        end        
        push!(coeff, c)
    end
    # println(idx,' ', coeff)
    Gurobi.add_constr!(model, idx, coeff, '=', 0.0)
    Gurobi.update_model!(model)
    # ADD.value(α)
    model
end

"""
    extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes,offset)

Extract algebraic decision diagram representing the distribution of a variable `var`, using a `cache` of ADDs and `scopes`.
"""
function extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes,offset)
    if haskey(cache, node) return cache[node] end
    if issum(spn[node])
        @assert length(spn[node].children) == 2
        low = extractADD!(cache,spn,spn[node].children[1],var,scopes,offset)
        high = extractADD!(cache,spn,spn[node].children[2],var,scopes,offset)
        γ = ADD.Node(Int(node),low,high)
        cache[node] = γ
        return γ
    elseif isprod(spn[node])
        for j in spn[node].children
            if var in scopes[j]
                γ = extractADD!(cache,spn,j,var,scopes,offset)
                cache[node] = γ
                return γ
            end
        end        
    elseif isa(spn[node], IndicatorFunction) # leaf
        @assert spn[node].scope == var
        stride = convert(Int, offset + spn[node].value)
        γ = ADD.Terminal(MLExpr(1.0,stride))
        cache[node] = γ
        return γ
    elseif isa(spn[node], CategoricalDistribution) # leaf
        @assert spn[node].scope == var
        expr = mapreduce(i -> MLE(spn[node].values[i], offset + i), +, 1:length(spn[node].values) )
        γ = ADD.Terminal(expr)
        cache[node] = γ
        return γ
    else
        @error "Unsupported node type: $(typeof(spn[node]))."
    end
end

end # module
