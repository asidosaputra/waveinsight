using Statistics, LinearAlgebra
using JUDI, SlimOptim, HDF5, SegyIO
using JSON


PATH_MODEL = "imaging_parameters/models/init_model.h5"
PATH_SHOT = "imaging_parameters/shots/real_data.sgy"
PATH_PARAMS = "imaging_parameters/fwi_params/inversion_params.json"

#loading params
params = JSON.parsefile(PATH_PARAMS)

#Kosongkan log
# Kosongkan isi file log di awal
open("imaging_parameters/logs/log.txt", "w") do f
    write(f, "")  # atau cukup biarkan blok kosong
end


# Load starting model
n,d,o, v0 = read(h5open("imaging_parameters/models/init_model.h5","r"), "n", "d", "o", "m")
v0 = v0 * 1e-3

m0 = 1f0 ./(v0).^2

model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
id_z_sea_water = Int32(floor(Float32(params["sea_water_base"]) / Float32(d[2]))) 


# Bound constraints
vmin = ones(Float32,model0.n) * (params["min_vel"]*1e-3)
vmax = ones(Float32,model0.n) * (params["max_vel"]*1e-3)

vmin[:,1:id_z_sea_water] .= v0[:,1:id_z_sea_water]   # keep water column fixed
vmax[:,1:id_z_sea_water] .= v0[:,1:id_z_sea_water]


# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read(PATH_SHOT)
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1], params["freq"]*1e-3)  
q = judiVector(src_geometry,wavelet)

# Optimization parameters
fevals = params["iteration"]

# Objective function for minConf library
count = 0
function objective_function(x, misfit=mse)
    model0.m .= reshape(x,model0.n);

    # fwi function value and gradient
    fval, grad = fwi_objective(model0, q, d_obs; misfit=misfit)
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1
    return fval, vec(grad.data)
end

# Bound projection
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), size(x))
if params["objective_function"] == "studentst"
    ϕ = x-> objective_function(x, studentst)
elseif params["objective_function"] == "mse"
    ϕ = x-> objective_function(x)
end


# Definisikan fungsi callback
function log_callback(state)
    curr_iteration = size(state.ϕ_trace)[1]
    
    i= (curr_iteration-1)
    if i > 0
        N = fevals
        
        open("imaging_parameters/logs/log.txt", "a") do io  # Append mode
            write(io, "Iteration $i of $N\n")
        end
        h5open("results/progress/current_results_$(i).h5", "w") do f
            f["slowness"] = state.x
            f["gradient"] = state.g
        end
    end
        
end

# Setel opsi SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)

# Jalankan optimasi dengan callback
solst = spg(ϕ, vec(m0), proj, options; callback=log_callback)

h5open("results/final/final_results.h5", "w") do f
    f["slowness"] = solst.x
    f["gradient"] = solst.g
end