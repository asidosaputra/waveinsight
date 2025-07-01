# # Source Independent Full Wave Form Inversion (SI-FWI)
# # Author: Asido Saputra Sigalingging and Ign Sonny Winardhie
# # Date: Bandung, December 2022

using JUDI, SlimOptim, LinearAlgebra, HDF5, JUDI, SegyIO, JSON, Statistics, Plots

function hanning_filter(N)
    n = range(0, N, step=1)
    sin.(2*pi*n/(2*N))
end

function mean_window_filter(w, data)
    mean(data .* reshape(w, length(w), 1), dims=2)
end

function conv!(X, H)
    N = size(X)[1] + size(H)[1] - 1
    N_pow = nextpow(2, N)
    
    f = zeros(N_pow, size(X)[2])
    g = zeros(N_pow, 1)

    f[1:size(X)[1], :] = X
    g[1:size(H)[1], :] = H

    id = Int(floor(size(H)[1]/2)) + 1

    @inbounds for i in 1:size(X)[2]
        a = fft(f[:, i]).*(fft(g[:,1]) / maximum(real(fft(g[:,1]))))
        b = real(ifft(a))

        X[:, i] = @view b[id:Int(id+size(X)[1]-1)]
    end
end

function trace_ref(params; cut = true)
    model  = params[:model]
    q = params[:q]
    d_obs = params[:d_obs]
    shot_id  = params[:shot_id]
    tmin_ref = params[:tmin_ref]
    tmax_ref = params[:tmax_ref]
    trc_id_ref = params[:trc_id_ref]
    num_trc = params[:num_trc]

    #calculate index training
    dt = calculate_dt(model)
    id_tlc = Int(floor(tmin_ref/dt) + 1) 
    id_thc = Int(floor(tmax_ref/dt) + 1)
    N  = id_thc-id_tlc

    #forward at specific geomtery
    F = judiModeling(model, q[shot_id].geometry, d_obs[shot_id].geometry)
    d_syn_ref = (F*q[shot_id]).data[1]

    #resample data
    d_syn_ref_resample = time_resample(d_syn_ref, q[shot_id].geometry, dt)[id_tlc:id_thc, trc_id_ref:(trc_id_ref+num_trc)]
    d_obs_ref_resample = time_resample(d_obs.data[shot_id], q[shot_id].geometry, dt)[id_tlc:id_thc, trc_id_ref:(trc_id_ref+num_trc)]

    w = hanning_filter(N)

    trace_syn_ref = mean_window_filter(w, d_syn_ref_resample)        
    trace_obs_ref = mean_window_filter(w, d_obs_ref_resample)
    
    return (trace_syn_ref, trace_obs_ref)
end 

function sifwi(dsyn, dobs, trace_syn_ref, trace_obs_ref)
    alpha = 0.01.* mean(dobs.^2)

    conv!(dsyn, trace_obs_ref)              #conv d_syn * (W dobs)
    conv!(dobs, trace_syn_ref)              #conv d_obs * (W dsyn)

    dsyn .= dsyn .- dobs                    #delta d w.o allocation memory
    dsyn .= dsyn ./ sqrt.(dsyn .^2 .+ alpha)

    cross_corr = dsyn[end:-1:1, 1:end]        #flip xcorr in conv methods
    conv!(cross_corr, trace_obs_ref)          #cross corelation fval (*) (W*dobs))

    return norm(dsyn), cross_corr[end:-1:1, 1:end]
end


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

shot_id = 1
t0_ref = 1
t0_ref = 500

sipar = Dict(
        :model     => model0,
        :q         => q,
        :d_obs     => d_obs,
        :shot_id   => 1,
        :tmin_ref    => 0.0,
        :tmax_ref    => 110.0,
        :trc_id_ref  => 10,
        :num_trc   => 5
    )




#############################################################################
###################### CHECK-1 ##############################################
#############################################################################

# CHECK FIRST GRADIENT
trace_syn_ref, trace_obs_ref = trace_ref(sipar)
misfit_sifwi = (dsyn, dobs) -> sifwi(dsyn, dobs, trace_syn_ref, trace_obs_ref)
# fval, grad = fwi_objective(model0, q, d_obs; misfit=misfit_sifwi)
plot(grad.data)

function objective_function(x, misfit=mse)
    model0.m .= reshape(x,model0.n);

    trace_syn_ref, trace_obs_ref = trace_ref(sipar)
    misfit_sifwi = (dsyn, dobs) -> sifwi(dsyn, dobs, trace_syn_ref, trace_obs_ref)

    # fwi function value and gradient
    fval, grad = fwi_objective(model0, q, d_obs; misfit=misfit_sifwi)
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1
    return fval, vec(grad.data)
end

# Bound projection
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), size(x))

ϕ = x->objective_function(x)

# FWI with SPG
m0 = vec(model0.m)
options = spg_options(verbose=1, maxIter=10, memory=3);
# @time solsifwi = spg(ϕ, m0, proj, options)