module SPN2MILP

# Implementation of SPN2MILP algorithm
import AlgebraicDecisionDiagrams
# Aliases
const ADD = AlgebraicDecisionDiagrams
const MLExpr = ADD.MultilinearExpression
import Gurobi

import SumProductNetworks
const SPN = SumProductNetworks
# import SumProductNetworks: 
#     Node, SumNode, ProductNode, LeafNode, CategoricalDistribution, IndicatorFunction, GaussianDistribution,
#     isleaf, isprod, issum, SumProductNetwork

export spn2milp

"""
    spn2milp(spn::SumProductNetwork, ordering=:dfs, params=nothing, multiplier=1.0)

Translates sum-product network `spn` into MAP-equivalent mixed-integer linear program.
Require that sum nodes have exactly two children.

# parameters
- `spn`: sum-product network.
- `ordering`: variablle elimination ordering: either a list of variables indices (integers) or the name of a heuristic (`:dfs`: topological depth-first order, `:deg`: min-degree, `:bfs`: topological bread-first order).
- `params`: dictionory of Gurobi solver parameters.
- `multipler`: scaling constant for improving numerical precision when dealing with small numbers (default: 1.0). 
"""
function spn2milp(spn::SPN.SumProductNetwork, ordering::Union{Symbol,Array{<:Integer}}=:dfs, params::Union{Nothing,Dict{String,Any}}=nothing, multiplier=1.0, query=nothing)    
    # obtain scope of every node
    scopes = SPN.scopes(spn)
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> SPN.issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node / latent variable
    buckets = Dict{Int,Array{ADD.DecisionDiagram{MLExpr}}}( i => [] for i in sumnodes ) 

    # Create optimization model (interacts with Gurobi.jl)
    env = Gurobi.Env()
    # Allow passing of parameters to solve
    if !isnothing(params)
        for (param,value) in params
            Gurobi.setparam!(env, param, value)
        end
    end
    # setparam!(env, "Method", 2)   # choose to use Barrier method
    # setparams!(env; IterationLimit=100, Method=1) # set the maximum iterations and choose to use Simplex method
     # creates an empty model ("milp" is the model name)
    model = Gurobi.Model(env, "milp", :maximize)
    ## Domain graph
    graph = Dict{Int,Set{Int}}( i => Set{Int}() for i in sumnodes )
    ## First obtain ADDs for manifest variables
    offset = 0 # offset to apply to variable indices at ADD leaves
    # each optimization variable has index = offset + value
    vdims = SPN.vardims(spn) # var id => no. of values
    potentials = ADD.DecisionDiagram{MLExpr}[]
    for var in sort(scopes[1])
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        # add it to pool
        push!(potentials, α)
        # update domain graph (connect variables in the ADD's scope) if min-degree ordering is used
        if ordering == :deg
            sc = map(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))
            for i in sc
                union!(graph[i],sc)
            end
        end
        # get (index of) bottom-most variable (highest id of a sum node)
        # id = maximum(sc)  
        # id = maximum(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        # # get index of lowest variable according to elimination ordering
        # i,id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))  
        # @assert length(buckets[id]) == 0 # TODO iterate until finding an empty bucket      
        # associate ADD to corresponding bucket
        # push!(buckets[id], α)
        # Create corresponding optimization variables for leaves (interacts with Gurobi)
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
        # add SOS1 constraint: exactly one value must be 1 (interacts with Gurobi)
        # idx = collect((offset+1):(offset+vdims[var]))
        # coeff = ones(Float64, length(idx))
        ## Changed: this is to be done when reading/processing query/evidence
        # Gurobi.add_sos!(model, :SOS1, idx, coeff)
        offset += vdims[var] # update start index for next variable
    end
    ndecvars = offset # record number of binary optimization variables
    ## Then build ADDs for sum nodes (latent variables)
    for id in sumnodes
        # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[id], α)
    end  
    # Find variable elimination sequence
    if isa(ordering,Symbol) # use heuristic
        if ordering == :bfs
            ordering = sort(sumnodes, rev=true) # eliminate variables in topological bfs order
        elseif ordering == :dfs
            # Remove variables in topological dfs order
            # First compute the number of parents for each node
            pa = zeros(length(spn))
            for (i,n) in enumerate(spn)
                if !SPN.isleaf(n)
                    for j in n.children            
                        pa[j] += 1
                    end
                end
            end
            @assert count(isequal(0), pa) == 1 "SumProductNetworks has more than one parentless node"
            root = findfirst(isequal(0),pa) # root is the single parentless node
            # Kanh's algorithm: collect node ids in topological DFS order
            stack = Int[ root ]
            sizehint!(stack, length(spn))
            ordering = Int[ ] # topo dfs order
            sizehint!(ordering, length(sumnodes))
            while !isempty(stack)
                n = pop!(stack) # remove from top of stack
                if !SPN.isleaf(spn[n])
                    if SPN.issum(spn[n])
                        # add to elimination ordering
                        push!(ordering, n)
                    end
                    for j in spn[n].children
                        pa[j] -= 1
                        if pa[j] == 0
                            push!(stack, j)
                        end
                    end
                end
            end
            @assert length(ordering) == length(sumnodes)
            reverse!(ordering)
        elseif ordering == :deg
            # TODO: Apply min-fill or min-degree heuristic to obtain better elimination ordering
            ordering = Int[]
            sizehint!(ordering, length(sumnodes))
            while !isempty(graph)
                # find minimum degree node -- break ties by depth/variable id
                deg, k = minimum( p -> (length(p[2]), -p[1]), graph )
                j = -k
                push!(ordering, j)
                # remove j and incident edges
                for k in graph[j]
                    setdiff!(graph[k], j)
                end
                delete!(graph, j)
            end 
            @assert length(ordering) == length(sumnodes)
        else
            @error "Unknown elimination heuristic $(ordering). Available heuristics are :dfs, :bfs and :deg."
        end
    end
    @assert length(ordering) == length(sumnodes) 
    vorder = Dict{Int,Int}() # ordering of elimination of each variable (inverse mapping)
    for i=1:length(ordering)
        vorder[ordering[i]] = i
    end      
    # Add ADDs to appropriate buckets
    for α in potentials
        # get index of first variable to be eliminated
        i, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))
        push!(buckets[id], α)
    end
    # release pool of ADDs to be collected by garbage collector
    potentials = nothing
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
        # Gurobi.add_cvar!(model, 0.0, 0.0, Inf) # is it worth adding lower bounds? upper bounds?
        Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # is it worth adding lower bounds? upper bounds?
        offset += 1 # increase opt var counter
        # println("continuous ", f)
        idx = [offset] # indices of variables in constraint
        coeff = [-1.0] # coefficients in linear constraint
        for (m,c) in e
            if length(m.vars) == 1
                push!(idx, m.vars[1])
            else # Linearize bilinear term w = x*y with x in [0, u] and y in {0,1}
                @assert length(m.vars) == 2
                # smaller of variables is binary one
                if m.vars[1] < m.vars[2]
                    y, x = m.vars
                else
                    x, y = m.vars
                end                
                @assert y <= ndecvars # must be decision variable
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
                Gurobi.add_constr!(model,[id, x], [1.0, -1.0], '<', 0.0)
                # w - u*y ≤ 0 
                Gurobi.add_constr!(model,[id, y], [1.0, -1.0], '<', 0.0)
                # w - u*y - x ≥ -u
                Gurobi.add_constr!(model,[id, y, x], [1.0, -1.0, -1.0], '>', -1.0)
            end
            push!(coeff, c)
        end
        # println(idx, coeff)
        Gurobi.add_constr!(model, idx, multiplier * coeff, '=', 0.0)
        # println("$f = $e")
        cache[e] = f
    end
    # Run variable elimination to generate constraints
    before = time_ns()
    α = ADD.Terminal(MLExpr(1.0))
    for i = 1:(length(ordering)-1)
        var = ordering[i] # variable to eliminate
        print("[$i/$(length(ordering))] ")
        printstyled("Eliminate: ", var; color = :light_cyan)
        α = α * reduce(*, buckets[var]; init = ADD.Terminal(MLExpr(1.0)))
        empty!(buckets[var]) # allow used ADDs to be garbage collected
        α = ADD.marginalize(α, var)     
        # Obtain copy with modified leaves and generate constraints (interacts with JUMP / Gurobi)
        α = ADD.apply(renameleaves, α)
        # β = ADD.apply(renameleaves, α)
        # For path decomposition, add "message" to next bucket to be processed
        # printstyled("-> $(ordering[i+1])\n"; color = :green)     
        # push!(buckets[ordering[i+1]], β)
        # Create corresponding optimization variables for leaves (interacts with JUMP / Gurobi)
        # println(β)
        # Print out constraint
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
        now = time_ns()
        etime = (now-before)/1e9
        printstyled(" [$(etime)s]\n"; color = :light_black)
        before = now
    end
    # Objective (last variable elimination)
    α = α * reduce(*, buckets[ordering[end]]; init = ADD.Terminal(MLExpr(1.0)))
    α = ADD.marginalize(α, ordering[end])
    @assert isa(α,ADD.Terminal)
    # Add equality constraint to represent objective
    Gurobi.add_cvar!(model, multiplier)
    offset += 1
    idx = [offset]
    coeff = [-1.0]
    for (m, c) in ADD.value(α)
        if length(m.vars) == 1
            push!(idx, m.vars[1])
        else # Linearize bilinear term w = x*y
            @assert length(m.vars) == 2
            # Assumes both variables have domain [0,1]
            # Might lead to unfeasible program if this is violated
            if haskey(bilinterms,m)
                id = bilinterms[m] # id of w
            else # add continuous variable w to problem with 0 ≤ w ≤ 1
                Gurobi.add_cvar!(model, 0.0, 0.0, Inf) # add new variable representing bilinear term
                # Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # add new variable representing bilinear term
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
    Gurobi.add_constr!(model, idx, multiplier * coeff, '=', 0.0)
    Gurobi.update_model!(model)
    # ADD.value(α)
    model
end

function spn2milp_q(spn::SPN.SumProductNetwork, query, evidence, ordering::Union{Symbol,Array{<:Integer}}=:dfs, params::Union{Nothing,Dict{String,Any}}=nothing, multiplier=1.0)    
    # obtain scope of every node
    scopes = SPN.scopes(spn)
    # Extract ADDs for each variable
    ## Colect ids of sum nodes
    sumnodes = filter(i -> SPN.issum(spn[i]), 1:length(spn))
    ## Create a bucket for each sum node / latent variable
    buckets = Dict{Int,Array{ADD.DecisionDiagram{MLExpr}}}( i => [] for i in sumnodes )
    # Create optimization model (interacts with Gurobi.jl)
    env = Gurobi.Env()
    # Allow passing of parameters to solve
    if !isnothing(params)
        for (param,value) in params
            Gurobi.setparam!(env, param, value)
        end
    end
    # creates an empty model ("milp" is the model name)
    model = Gurobi.Model(env, "milp", :maximize)
    ## Domain graph
    graph = Dict{Int,Set{Int}}( i => Set{Int}() for i in sumnodes )
    ## First obtain ADDs for manifest variables
    offset = 0 # offset to apply to variable indices at ADD leaves
    # each optimization variable has index = offset + value
    vdims = SPN.vardims(spn) # var id => no. of values
    potentials = ADD.DecisionDiagram{MLExpr}[]
    for var in sort(collect(query))
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        # add it to pool
        push!(potentials, α)
        # update domain graph (connect variables in the ADD's scope) if min-degree ordering is used
        if ordering == :deg
            sc = map(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))
            for i in sc
                union!(graph[i],sc)
            end
        end            
        for i=1:vdims[var]
            # syntax: model, coefficient in objective
            Gurobi.add_bvar!(model, 0.0)
        end
        # add SOS1 constraint: exactly one value must be 1 (interacts with Gurobi)
        idx = collect((offset+1):(offset+vdims[var]))
        coeff = ones(Float64, length(idx))
        Gurobi.add_sos!(model, :SOS1, idx, coeff)
        offset += vdims[var] # update start index for next variable
    end
    for var in sort!(collect(keys(evidence)))
        # Extract ADD for variable var
        α = ADD.reduce(extractADD!(Dict{Int,ADD.DecisionDiagram{MLExpr}}(), spn, 1, var, scopes, offset))
        # add it to pool
        push!(potentials, α)
        # update domain graph (connect variables in the ADD's scope) if min-degree ordering is used
        if ordering == :deg
            sc = map(ADD.index, Base.filter(n -> isa(n,ADD.Node), collect(α)))
            for i in sc
                union!(graph[i],sc)
            end
        end                    
        for i=1:vdims[var]
            # syntax: model, coefficient in objective
            Gurobi.add_bvar!(model, 0.0)
        end
        # add constraints for evidence
        for j = 1:vdims[var]
            if j == evidence[var]
                c = 1.0
            else
                c = 0.0
            end
            Gurobi.add_constr!(model, [ offset + j ], [1.0], '=', c)
        end
        offset += vdims[var] # update start index for next variable
    end    
    ndecvars = offset # record number of binary optimization variables
    ## Then build ADDs for sum nodes (latent variables)
    for id in sumnodes
        # construct ADD
        α = ADD.Node(id,MLExpr(spn[id].weights[1]),MLExpr(spn[id].weights[2]))
        # associate ADD to corresponding bucket
        push!(buckets[id], α)
    end  
    # Find variable elimination sequence
    if isa(ordering,Symbol) # use heuristic
        if ordering == :bfs
            ordering = sort(sumnodes, rev=true) # eliminate variables in topological bfs order
        elseif ordering == :dfs
            # Remove variables in topological dfs order
            # First compute the number of parents for each node
            pa = zeros(length(spn))
            for (i,n) in enumerate(spn)
                if !SPN.isleaf(n)
                    for j in n.children            
                        pa[j] += 1
                    end
                end
            end
            @assert count(isequal(0), pa) == 1 "SumProductNetworks has more than one parentless node"
            root = findfirst(isequal(0),pa) # root is the single parentless node
            # Kanh's algorithm: collect node ids in topological DFS order
            stack = Int[ root ]
            sizehint!(stack, length(spn))
            ordering = Int[ ] # topo dfs order
            sizehint!(ordering, length(sumnodes))
            while !isempty(stack)
                n = pop!(stack) # remove from top of stack
                if !SPN.isleaf(spn[n])
                    if SPN.issum(spn[n])
                        # add to elimination ordering
                        push!(ordering, n)
                    end
                    for j in spn[n].children
                        pa[j] -= 1
                        if pa[j] == 0
                            push!(stack, j)
                        end
                    end
                end
            end
            @assert length(ordering) == length(sumnodes)
            reverse!(ordering)
        elseif ordering == :deg
            # TODO: Apply min-fill or min-degree heuristic to obtain better elimination ordering
            ordering = Int[]
            sizehint!(ordering, length(sumnodes))
            while !isempty(graph)
                # find minimum degree node -- break ties by depth/variable id
                deg, k = minimum( p -> (length(p[2]), -p[1]), graph )
                j = -k
                push!(ordering, j)
                # remove j and incident edges
                for k in graph[j]
                    setdiff!(graph[k], j)
                end
                delete!(graph, j)
            end 
            @assert length(ordering) == length(sumnodes)
        else
            @error "Unknown elimination heuristic $(ordering). Available heuristics are :dfs, :bfs and :deg."
        end
    end
    @assert length(ordering) == length(sumnodes) 
    vorder = Dict{Int,Int}() # ordering of elimination of each variable (inverse mapping)
    for i=1:length(ordering)
        vorder[ordering[i]] = i
    end      
    # Add ADDs to appropriate buckets
    for α in potentials
        # get index of first variable to be eliminated
        i, id = minimum(n -> (vorder[ADD.index(n)],ADD.index(n)), Base.filter(n -> isa(n,ADD.Node), collect(α)))
        push!(buckets[id], α)
    end
    # release pool of ADDs to be collected by garbage collector
    potentials = nothing
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
        # Gurobi.add_cvar!(model, 0.0, 0.0, Inf) # is it worth adding lower bounds? upper bounds?
        Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # is it worth adding lower bounds? upper bounds?
        offset += 1 # increase opt var counter
        # println("continuous ", f)
        idx = [offset] # indices of variables in constraint
        coeff = [-1.0] # coefficients in linear constraint
        constant = 0.0
        for (m,c) in e
            if length(m.vars) == 0
                constant += c
            elseif length(m.vars) == 1
                push!(idx, m.vars[1])
                push!(coeff, c)
            else # Linearize bilinear term w = x*y with x in [0, u] and y in {0,1}
                @assert length(m.vars) == 2 "Found multilinear monomial: $m"
                # smaller of variables is binary one
                if m.vars[1] < m.vars[2]
                    y, x = m.vars
                else
                    x, y = m.vars
                end                
                @assert y <= ndecvars "Both variables in bilinear monomial $m are continuous (binary variables < $ndecvars)." # must be decision variable
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
                Gurobi.add_constr!(model,[id, x], [1.0, -1.0], '<', 0.0)
                # w - u*y ≤ 0 
                Gurobi.add_constr!(model,[id, y], [1.0, -1.0], '<', 0.0)
                # w - u*y - x ≥ -u
                Gurobi.add_constr!(model,[id, y, x], [1.0, -1.0, -1.0], '>', -1.0)
                push!(coeff, c)
            end
        end
        # println(idx, coeff)
        Gurobi.add_constr!(model, idx, multiplier * coeff, '=', constant)
        # println("$f = $e")
        cache[e] = f
    end
    # Run variable elimination to generate constraints
    before = time_ns()
    α = ADD.Terminal(MLExpr(1.0))
    for i = 1:(length(ordering)-1)
        var = ordering[i] # variable to eliminate
        print("[$i/$(length(ordering))] ")
        printstyled("Eliminate: ", var; color = :light_cyan)
        α = α * reduce(*, buckets[var]; init = ADD.Terminal(MLExpr(1.0)))
        empty!(buckets[var]) # allow used ADDs to be garbage collected
        α = ADD.marginalize(α, var)     
        # Obtain copy with modified leaves and generate constraints (interacts with JUMP / Gurobi)
        α = ADD.apply(renameleaves, α)   
        now = time_ns()
        etime = (now-before)/1e9
        printstyled(" [$(etime)s]\n"; color = :light_black)
        before = now
    end
    # Objective (last variable elimination)
    α = α * reduce(*, buckets[ordering[end]]; init = ADD.Terminal(MLExpr(1.0)))
    α = ADD.marginalize(α, ordering[end])
    @assert isa(α,ADD.Terminal)
    # Add equality constraint to represent objective
    Gurobi.add_cvar!(model, multiplier)
    offset += 1
    idx = [offset]
    coeff = [-1.0]
    for (m, c) in ADD.value(α)
        if length(m.vars) == 1
            push!(idx, m.vars[1])
        else # Linearize bilinear term w = x*y
            @assert length(m.vars) == 2
            # Assumes both variables have domain [0,1]
            # Might lead to unfeasible program if this is violated
            if haskey(bilinterms,m)
                id = bilinterms[m] # id of w
            else # add continuous variable w to problem with 0 ≤ w ≤ 1
                Gurobi.add_cvar!(model, 0.0, 0.0, Inf) # add new variable representing bilinear term
                # Gurobi.add_cvar!(model, 0.0, 0.0, 1.0) # add new variable representing bilinear term
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
    Gurobi.add_constr!(model, idx, multiplier * coeff, '=', 0.0)
    Gurobi.update_model!(model)
    model
end

"""
    extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SumProductNetwork,node::Integer,var::Integer,scopes,offset)

Extract algebraic decision diagram representing the distribution of a variable `var`, using a `cache` of ADDs and `scopes`.
`offset` gives the starting index of variables at the leaves.
"""
function extractADD!(cache::Dict{Int,ADD.DecisionDiagram{MLExpr}},spn::SPN.SumProductNetwork,node::Integer,var::Integer,scopes,offset)
    # @assert var in scopes[node]
    if haskey(cache, node) return cache[node] end
    if SPN.issum(spn[node])
        @assert length(spn[node].children) == 2
        low = extractADD!(cache,spn,spn[node].children[1],var,scopes,offset)
        high = extractADD!(cache,spn,spn[node].children[2],var,scopes,offset)
        γ = ADD.Node(Int(node),low,high)
        cache[node] = γ
        return γ
    elseif SPN.isprod(spn[node])
        for j in spn[node].children
            if var in scopes[j]
                γ = extractADD!(cache,spn,j,var,scopes,offset)
                cache[node] = γ
                return γ
            end
        end
        cache[node] = ADD.Terminal(MLExpr(1.0))
        return cache[node]
        # error("$var is not in node $(node)'s scope. $(scopes[node])")
    elseif isa(spn[node], SPN.IndicatorFunction) # leaf
        @assert spn[node].scope == var
        stride = convert(Int, offset + spn[node].value)
        γ = ADD.Terminal(MLExpr(1.0,stride))
        cache[node] = γ
        return γ
    elseif isa(spn[node], SPN.CategoricalDistribution) # leaf
        @assert spn[node].scope == var
        expr = mapreduce(i -> MLE(spn[node].values[i], offset + i), +, 1:length(spn[node].values) )
        γ = ADD.Terminal(expr)
        cache[node] = γ
        return γ
    else
        @error "Unsupported node type: $(typeof(spn[node]))."
    end
    @error "Reached end of cases, something went wrong."
end

end # module
