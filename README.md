Fachgespräch at VAW, 13.01.2023
==============================

Presentation material for the talk "Introduction to differential modelling: demystifying the adjoint method".

## Automatic notebook generation

The presentation slides and the demo notebook are self-contained in a Jupyter notebook [vaw-fachgesprach-2023.ipynb](vaw-fachgesprach-2023.ipynb) that can be auto-generated using literate programming by deploying the [deploy.jl](deploy.jl) script.

To reproduce:
1. Clone this git repo
2. Open Julia and resolve/instantiate the project
```julia-repl
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
```

3. Run the deploy script
```julia-repl
julia> using Literate

julia> include("deploy.jl")
```
4. Then using IJulia, you can launch the notebook and get it displayed in your web browser:
```julia-repl
julia> using IJulia

julia> notebook(dir=pwd())
```
_To view the notebook as slide, you need to install the [RISE](https://rise.readthedocs.io/en/stable/installation.html) plugin_
