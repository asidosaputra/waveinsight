import streamlit as st
import tempfile
from utils.utils import *
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import atexit
import time
import h5py
import io
import json
import subprocess


def delete_file(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Gagal menghapus file: {e}")

def cleanup_temporary_files():
    if "temporary_files" in st.session_state:
        for path in st.session_state.temporary_files:
            delete_file(path)

# Daftarkan cleanup saat Streamlit shutdown
atexit.register(cleanup_temporary_files)

st.set_page_config(layout="wide")
# Data awal
default_byte_location = {
    "ffid": 9,
    "trace_number": 13,
    "sx": 73,
    "sy": 77,
    "gx": 81,
    "gy": 85,
}

# --- Sidebar Navigation ---
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih Fungsi",
    ["1. Prepare Shot", "2. Prepare Model", "3. Imaging", "4. Visualize"]
)

# --- Halaman: Prepare Shot ---
if menu == "1. Prepare Shot":
    if "data_gather" not in st.session_state:
        st.session_state.data_gather = None
        st.session_state.dt = None

    if "data_header" not in st.session_state:
        st.session_state.data_header = None

    st.title("Prepare Shot Data and Geometry")
    st.write("Modul for preparing shot data and geometry")
    
    with st.expander("Upload Shot"):
        col1, col2 = st.columns([1, 3])
        with col1:
            uploaded_shot = st.file_uploader("Upload file shot (.sgy, .segy)", type=["sgy", "segy"])
            if uploaded_shot is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sgy") as tmp_file:
                    tmp_file.write(uploaded_shot.read())
                    tmp_filepath = tmp_file.name
                    st.success(f"File berhasil diunggah: {uploaded_shot.name}")

    if uploaded_shot is not None:
        with st.expander("Loading and Header Editor"):
            col1, col2 = st.columns([3, 2])
            with col1: 
                load_option = st.selectbox("Header Byte Location", options=["Default", "Custom"])
                is_load = False
                if load_option == "Custom":
                    # Buat form
                    is_custom_ready = False
                    with st.form("edit_byte_form"):
                        keys = list(default_byte_location.keys())
                        cols = st.columns(len(keys))  # Buat kolom horizontal sebanyak jumlah key

                        new_byte_location = {}
                        for col, key in zip(cols, keys):
                            with col:
                                new_val = st.number_input(f"{key}", value=default_byte_location[key], min_value=0, max_value=239, step=1)
                                new_byte_location[key] = new_val
                        
                        submitted_byte =  st.form_submit_button("Loding Data", type="primary")
                        
                        if submitted_byte:
                            try:
                                trace_header, gathers, dt = read_segy(tmp_filepath, new_byte_location)
                                st.session_state.data_gather = gathers
                                st.session_state.data_header = trace_header
                                st.session_state.dt = dt*1e-6
                                st.success("Loading data Succesfully")

                            except:
                                st.error("Loading data fail!")
                else:
                    if st.button("Loading Data", type="primary"):
                        try:
                            trace_header, gathers, dt = read_segy(tmp_filepath)
                            st.session_state.data_gather = gathers
                            st.session_state.data_header = trace_header
                            st.session_state.dt = dt*1e-6

                            st.success("Loading data Succesfully")

                        except:
                            st.error("Loading data fail!")

            col1, col2 = st.columns([3, 2])
            with col1:    
                if st.session_state.data_header is not None:
                    st.session_state.data_header = st.data_editor(st.session_state.data_header)
                    
            with col2:    
                if st.session_state.data_header is not None:
                    uploaded_header_df = st.file_uploader("Upload Header", type=["csv"]) 
                    if uploaded_header_df is not None:
                        # st.write()
                        # st.write()
                        if st.button("Update"):
                            df = pd.read_csv(uploaded_header_df)
                            st.session_state.data_header = df
                    else:
                        st.button("Update", disabled=True, key="btn_hdr_disable")
                        
            if st.session_state.data_header is not None:
                geom_plot = get_geom_uniq(st.session_state.data_header)
                fig = go.Figure(data=[
                    go.Scatter(x=geom_plot[3], y=geom_plot[4], mode='markers',  marker_symbol="triangle-down", marker_size=10, name='Receiver'),
                    go.Scatter(x=geom_plot[1], y=geom_plot[2], mode='markers', marker_symbol="star", marker_size=10, name='Source', marker_color="red")
                    
                ])
                fig.update_layout(
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    xaxis=dict(tickformat=".0f")

                )
                st.divider()
                st.plotly_chart(fig)

        if st.session_state.data_header is not None:
            with st.expander("Prepare Shot Data"):
                # Sorted FFID list
                ffid_list = sorted(st.session_state.data_gather.keys())
                # Simpan indeks di session_state
                if "ffid_index" not in st.session_state:
                    st.session_state.ffid_index = 0

                # Navigasi
                col11, col12, col13 = st.columns([1, 1, 8])
                with col11:
                    if st.button("⬅️ Previous", use_container_width=True) and st.session_state.ffid_index > 0:
                        st.session_state.ffid_index -= 1
                with col12:
                    if st.button("Next ➡️", use_container_width=True) and st.session_state.ffid_index < len(ffid_list) - 1:
                        st.session_state.ffid_index += 1
                # Ambil gather saat ini
                current_ffid = ffid_list[st.session_state.ffid_index]
                gather = st.session_state.data_gather[current_ffid]
                
                col21, col22 = st.columns([2, 8])
                with col21:
                    filter_selected = st.selectbox("Apply Filter", options=["None", "Bandpass Filter"])

                st.divider()
            
                col31, col32, col33 = st.columns([2, 2, 2])
                if filter_selected == "None":
                    with col31:
                        _,freq, spectrum = extract_amplitude_and_spectrum(gather.T, st.session_state.dt)
                        df = pd.DataFrame({"Frequency (Hz)": freq, "Amplitude": spectrum})
                        fig = px.line(df, x="Frequency (Hz)", y="Amplitude")
                        fig.update_layout(template="plotly_white", height=500)
                        st.plotly_chart(fig, use_container_width=True)

                    with col32:
                        # Plot pakai Plotly
                        fig = px.imshow(
                            gather,
                            aspect='auto',
                            origin='upper',
                            title=f"FFID: {current_ffid} | Trace Number: {gather.shape[1]}",
                            color_continuous_scale='gray',
                            labels={"x": "Receiver", "y": "Time [ms]"},
                        )
                        fig.update_layout(
                            height=500,
                            title_x = 0.2,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if "lowcut_gather" not in st.session_state:
                        st.session_state.lowcut_gather = 1

                    if "highcut_gather" not in st.session_state:
                        st.session_state.highcut_gather = 60.0

                    # _,freq, spectrum = extract_amplitude_and_spectrum(gather.T, st.session_state.dt)
                    
                    with col22:
                        col11_22, col12_22, col13_22, col14_22 = st.columns([1, 1, 1, 4])
                        with col11_22:
                            lowcut = st.number_input("Lowcut Frequency (Hz)", min_value=0.1, value=1.0, step=1.0)
                        with col12_22:
                            highcut = st.number_input("Highcut Frequency (Hz)", min_value=lowcut, value=100.0, step=1.0)
                    with col21:
                        st.write("")
                        st.button("Apply Filter", key="btn_apply_filter", use_container_width=True, type="primary")

                    if st.session_state.btn_apply_filter == True:
                        st.session_state.lowcut_gather = lowcut
                        st.session_state.highcut_gather = highcut

                    gather_filtered = butter_bandpass_filter(gather, st.session_state.dt, st.session_state.lowcut_gather, st.session_state.highcut_gather, order=4)

                    with col31:
                        _,freq, spectrum = extract_amplitude_and_spectrum(gather.T, st.session_state.dt)
                        _,freq, spectrum_filter = extract_amplitude_and_spectrum(gather_filtered.T, st.session_state.dt)
                        # Inisialisasi plot
                        fig = go.Figure()

                        # Tambahkan garis spektrum original
                        fig.add_trace(go.Scatter(
                            x=freq, y=spectrum,
                            mode='lines',
                            name='Original Spectrum',
                            line=dict(color='blue')
                        ))

                        # Tambahkan garis spektrum filter
                        fig.add_trace(go.Scatter(
                            x=freq, y=spectrum_filter,
                            mode='lines',
                            name='Filtered Spectrum',
                            line=dict(color='red', dash='dash')
                        ))

                        # Layout dan label
                        fig.update_layout(
                            title='Spectrum Comparison',
                            xaxis_title='Frequency (Hz)',
                            yaxis_title='Amplitude',
                            template='plotly_white'
                        )

                        # Tampilkan plot di Streamlit
                        st.plotly_chart(fig, use_container_width=True)

                    with col32:
                        # Plot pakai Plotly
                        fig = px.imshow(
                            gather,
                            aspect='auto',
                            origin='upper',
                            title=f"FFID: {current_ffid} | Trace Number: {gather.shape[1]} | Original",
                            color_continuous_scale='gray',
                            labels={"x": "Receiver", "y": "Time [ms]"},
                        )
                        fig.update_layout(
                            height=500,
                            title_x = 0.2,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col33:
                        # Plot pakai Plotly
                        fig = px.imshow(
                            gather_filtered,
                            aspect='auto',
                            origin='upper',
                            title=f"FFID: {current_ffid} | Trace Number: {gather.shape[1]} | Filtered",
                            
                            color_continuous_scale='gray',
                            labels={"x": "Receiver", "y": "Time [ms]"},
                        )
                        fig.update_layout(
                            height=500,
                            title_x = 0.2,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with st.expander("Downdload Data"):
                
                # Simulasi file hasil upload atau proses
                if "download_ready" not in st.session_state:
                    st.session_state.download_ready = False
                    st.session_state.file_bytes = None
                
                # Tombol untuk memulai proses
                if st.button("🚀 Processing SEGY"):
                    # Baca hasilnya dalam bentuk bytes
                    with st.spinner("Wait for it...", show_time=True):
                        time.sleep(3)
                        if filter_selected == "None":
                            data_out = st.session_state.data_gather
                        else:
                            gathers_all_filter = filter_all_shot_data(st.session_state.data_gather, st.session_state.dt, st.session_state.lowcut_gather, st.session_state.highcut_gather, order=4)
                            data_out = gathers_all_filter

                        df = pd.DataFrame(st.session_state.data_header)
                        set_header_and_data(tmp_filepath, data_out, df, byte_location=None)
                    
                    with open(tmp_filepath, "rb") as f:
                        st.session_state.file_bytes = f.read()
                        st.session_state.download_ready = True

                if st.session_state.download_ready:
                    st.download_button(
                        label="📥 Download File",
                        data=st.session_state.file_bytes,
                        file_name="shot_data_processed.sgy",
                        mime="application/octet-stream"
                    )   
                
    else:
        print("Masing Kosong")

# --- Halaman: Prepare Model ---
elif menu == "2. Prepare Model":
    st.title("Prepare Model")

    if "initial_model" not in st.session_state:
        st.session_state.initial_model = None

    col11, col12 = st.columns([1, 3])
    with col11:
        selected_type_model = st.selectbox("How would you like to create the initial model??", ["Generate", "Upload"])

    print(selected_type_model)
    if selected_type_model == "Generate":
        # Form untuk parameter model
        with st.expander("Cell Model Parameters"):
            st.write("Generate Smooth Model")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                nx = st.number_input("nx (Number of Cells in X)", min_value=1, max_value=2000, step=1)
                nz = st.number_input("nz (Number of Cells in Z)", min_value=1, max_value=2000, step=1)

            with col2:
                dx = st.number_input("dx (Cell Size in X) [m]", min_value=1.0, max_value=1000.0, step=0.5)
                dz = st.number_input("dz (Cell Size in Z) [m]", min_value=1.0, max_value=1000.0, step=0.5)

            with col3:
                ox = st.number_input("ox (Origin X) [m]", min_value=0.0,)
                oz = st.number_input("oz (Origin Z) [m]", min_value=0.0,)

            with col4:
                vmin = st.number_input("vmin (Min Velocity - Top Layer) [m/s]", min_value=1.0, max_value=10000.0, step=0.5)
                vmax = st.number_input("vmax (Max Velocity - Bottom Layer) [m/s]", min_value=1.0, max_value=10000.0, step=0.5)

            submitted_gen_model = st.button("Generate", type="primary")

        if submitted_gen_model:
            vel_1D_z = np.linspace(vmin, vmax, nz)
            vel_data = np.repeat([vel_1D_z], nx, axis=0).T
            st.session_state.initial_model = {"n":(nx,nz), "d":(dx,dz), "o":(ox,oz), "m" : vel_data}
        
        # Memastikan session state untuk header model
        if "QC_header_model" not in st.session_state:
            st.session_state.QC_header_model = None

        # Upload file header dan plot
        with st.expander("QC Model and Geometry"):
            upload_header_df = st.file_uploader("Upload Source-Group Header (.csv)")
            if upload_header_df is not None:
                if st.button("Plot Header", type="primary"):
                    st.session_state.QC_header_model = get_geom_uniq(pd.read_csv(upload_header_df))
            else:
                st.button("Plot Header", key="plot_geom_header", disabled=True)
                st.session_state.QC_header_model = None

            # Jika model header sudah ada, ambil data X
            if st.session_state.QC_header_model is not None:
                x_src = st.session_state.QC_header_model[1]
                x_rcv = st.session_state.QC_header_model[3]
                x_geom = np.concatenate((x_src, x_rcv))  # Gabung source & receiver

            # Jika model sudah ada, buat koordinat X
            if st.session_state.initial_model is not None:
                x_model = np.arange(0, st.session_state.initial_model["n"][0]) * st.session_state.initial_model["d"][0]
                x_model = st.session_state.initial_model["o"][0] + x_model

            # Hitung rentang X gabungan dengan pengecekan apakah ada data
            x_all = []

            if st.session_state.get("QC_header_model") is not None:
                x_all.append(np.min(x_geom))
                x_all.append(np.max(x_geom))

            if st.session_state.get("init_model") is not None:
                x_all.append(np.min(x_model))
                x_all.append(np.max(x_model))

            # Cegah error saat x_all kosong
            if x_all:
                xlim = [min(x_all), max(x_all)]
            else:
                xlim = [0, 1]  # Atau bisa None atau range default lain
                st.warning("Data koordinat X belum tersedia. Harap upload header atau buat model terlebih dahulu.")

            # Plot Geometry
            if st.session_state.QC_header_model is not None:
                fig = go.Figure(data=[
                    go.Scatter(x=x_rcv, y=st.session_state.QC_header_model[4], mode='markers', marker_symbol="triangle-down", marker_size=10, name='Receiver'),
                    go.Scatter(x=x_src, y=st.session_state.QC_header_model[2], mode='markers', marker_symbol="star", marker_size=10, name='Source', marker_color="red")
                ])
                fig.update_layout(
                    title="Geometry Plot",
                    title_x=0.5,
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    xaxis=dict(range=xlim, tickformat=".0f")  # Set xlim di sini
                )

                st.plotly_chart(fig)

            # Plot Model Velocity
            if st.session_state.initial_model is not None:
                st.divider()

                # Z coordinate
                z_model = np.arange(0, st.session_state.initial_model["n"][1]) * st.session_state.initial_model["d"][1]
                z_model = st.session_state.initial_model["o"][1] + z_model

                # Heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.initial_model["m"],
                    x=x_model,
                    y=z_model,
                    colorscale='Jet',
                    colorbar=dict(title='Velocity (m/s)')
                ))

                fig.update_yaxes(
                    autorange="reversed",
                    title="Depth (m)",
                    tickformat=".1f"
                )
                fig.update_xaxes(
                    title="Distance X (m)",
                    range=xlim,  # Set xlim sama dengan geometry plot
                    tickformat=".1f"
                )

                fig.update_layout(
                    title="2D Velocity Model",
                    height=600,
                    title_x=0.5,
                )

                st.plotly_chart(fig, use_container_width=True)

    elif selected_type_model == "Upload":
        print("Masuk sini")
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_format_upload_model = st.selectbox("Choose format of initial model!", [".sgy", ".h5", ".txt"])

    with st.expander("Downdload"):
        if st.session_state.initial_model is not None:
            if st.button("Process Data"):
                st.write("Downdload")
                model_out = st.session_state.initial_model

                # Simpan ke dalam file HDF5 di memory (pakai io.BytesIO)
                h5_buffer = io.BytesIO()
                with h5py.File(h5_buffer, 'w') as h5file:
                    for key, value in model_out.items():
                        h5file.create_dataset(key, data=value)

                h5_buffer.seek(0)

                # Tombol download
                st.download_button(
                    label="Download Model as .h5",
                    data=h5_buffer,
                    file_name="model_data.h5",
                    mime="application/octet-stream"
                )
                
# --- Halaman: Imaging ---
elif menu == "3. Imaging":
    st.title("Imaging")
    st.write("What type of imaging do you prefer?")
    selected_imaging = st.selectbox("Type of Imaging", ["FWI", "Migration"])

    if selected_imaging == "FWI":
        if "fwi_parameters" not in st.session_state:
            st.session_state.fwi_parameters = None
        
        if "init_model_imaging" not in st.session_state:
            st.session_state.init_model_imaging = None
        
        PATH_MODEL = "imaging_parameters/models/init_model.h5"
        PATH_SHOT = "imaging_parameters/shots/real_data.sgy"
        PATH_PARAMS = "imaging_parameters/fwi_params/inversion_params.json"

        with st.expander("Initial model and Shot Data"):
            init_model_imaging = st.file_uploader("Upload Initial Model  (.h5)", type=[".h5"])
            shot_data_imaging = st.file_uploader("Shot Data  (.segy)", type=["sgy", "segy"])

            if init_model_imaging is not None and shot_data_imaging is not None:
                if st.button("Saving Data", type="primary"):
                    try:
                        # Simpan file ke server
                        with open(PATH_MODEL, "wb") as f:
                            f.write(init_model_imaging.getbuffer())
                        
                        with h5py.File(PATH_MODEL, 'r') as f:
                            st.session_state.init_model_imaging = {"n":np.array(f["n"]), "d":np.array(f["d"]), "o":np.array(f["o"]), "m":np.array(f["m"])}
                    except:
                        st.error("Saving Init Model Fail!")   
            
                    # Simpan file ke server
                    try:
                        with open(PATH_SHOT, "wb") as f:
                            f.write(shot_data_imaging.getbuffer())
                            st.success("Saving Model and Shot Data Succes..")
                    except:
                        st.error("Saving Data Fail!")
                    
            else:
                st.error("Please upload model and shot")

        if st.session_state.init_model_imaging is not None:
            with st.expander("Source Wavelet"):
                selected_wavelet = st.selectbox("How do you prefer modelling wavelet?", ["Generate", "Upload"])

                if selected_wavelet == "Generate":
                    freq = st.number_input("Ricker Frequency (Hz)", min_value=1.0, max_value=200.0, step=0.5 )


            with st.expander("Full Wave Inversion Parameters"):
                col11, col12, col13 = st.columns([1, 1, 1])
                with col11:
                    selected_optimize = st.selectbox("Optimization Methods", ["SPG"])
                with col12:
                    selected_objective = st.selectbox("Objective Function", ["Studentst", "MSE", "SI_FWI"])

                with col13:
                    max_iter = st.number_input("Max Iteration", min_value=1, max_value=100, step=1)
                
                col21, col22, col23 = st.columns([1,1,1])
                # with st.container():
                with col21:
                    bound_vel_min = st.number_input("Min Velocity Bound (m/s)", min_value=100.00, max_value=10000.0, step=10.0)
                with col22:
                    bound_vel_max = st.number_input("Max Velocity Bound (m/s)", min_value=100.00, max_value=10000.0, step=10.0)
                with col23:
                    sea_water_base = st.number_input("Bottom Sea Water Depth (m)", min_value=1.00, max_value=10000.0, step=10.0)

                if selected_objective =="SI_FWI":
                    if "header_data" not in st.session_state:
                        st.session_state.header_data, _ = read_header_only(PATH_SHOT)
                    else:
                        st.session_state.header_data, _ = read_header_only(PATH_SHOT)
            
                    ffid_uniq = np.unique(st.session_state.header_data["ffid"])

                    with col21:
                        shot_id = st.selectbox("Shot FFID", ffid_uniq)
                        
                        t_min = st.number_input("Min Time (ms)", min_value=0, max_value=10000, step=1)

                    with col22:
                        trace_id = st.number_input("Trace ID", min_value=1, max_value=10000, step=1)
                        t_max = st.number_input("Max Time (ms)", min_value=0, max_value=10000, step=1)
                        

                    with col23:
                        trace_num = st.number_input("Number Trace", min_value=1, max_value=10000, step=1)
                else:
                    shot_id = None
                    t_min = None
                    t_max = None
                    trace_id = None
                    trace_num = None
                        
                # Inisialisasi session_state jika belum ada
                if "julia_process" not in st.session_state:
                    st.session_state.julia_process = None

                if st.button("RUN FWI", type="primary"):
                    st.write("Loading FWI Parameteres..")
                    try:
                        st.session_state.fwi_parameters = {"freq":freq, "iteration":max_iter, 
                                                           "objective_function":selected_objective.lower(), 
                                                           "min_vel":bound_vel_min, "max_vel":bound_vel_max, 
                                                           "sea_water_base":sea_water_base,
                                                           "shot_id": shot_id,
                                                           "trc_id_ref":trace_id,
                                                           "num_trace":trace_num,
                                                           "tmin_ref":t_min,
                                                           "tmax_ref":t_max}
                        
                        with open(PATH_PARAMS, 'w') as f:
                            json.dump(st.session_state.fwi_parameters, f)

                        st.success("Succes Saving Inversion Parameters")
                        st.write("Inversion in Progress..")

                        total_iters = st.session_state.fwi_parameters["iteration"]
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        log_file = "imaging_parameters/logs/log.txt"
                        try:
                            # Hapus log lama
                            if os.path.exists(log_file):
                                os.remove(log_file)

                            process = subprocess.Popen(
                                ["/opt/julia-1.10.9/bin/julia", "julia_scripts/run_fwi.jl"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )

                            st.session_state.julia_process = process  # Simpan ke session_state

                            # Monitoring
                            iter_count = 0
                            max_wait = 10000
                            start_time = time.time()

                            while True:
                                if time.time() - start_time > max_wait:
                                    process.kill()
                                    raise TimeoutError("Proses terlalu lama.")

                                # Jika user sudah tekan tombol stop (lihat bawah)
                                if st.session_state.get("stop_requested", False):
                                    process.terminate()  # atau .kill() kalau perlu paksa
                                    st.warning("Proses dihentikan oleh pengguna.")
                                    break

                                if os.path.exists(log_file):
                                    with open(log_file, "r") as f:
                                        lines = f.readlines()
                                        iter_count = len(lines)

                                    percent = iter_count / total_iters
                                    progress_bar.progress(min(percent, 1.0))
                                    status_text.text(f"Iterasi: {iter_count} dari {total_iters}")

                                    if iter_count >= total_iters:
                                        break

                                time.sleep(0.2)

                            # Ambil output
                            stdout, stderr = process.communicate()
                            if process.returncode != 0:
                                raise RuntimeError(f"Proses Julia gagal:\n{stderr}")

                            if not st.session_state.get("stop_requested", False):
                                st.success("Optimasi selesai!")

                        except FileNotFoundError:
                            st.error("Interpreter Julia tidak ditemukan.")
                        except TimeoutError as te:
                            st.error(f"Timeout: {te}")
                        except RuntimeError as re:
                            st.error(f"Error: {re}")
                        except Exception as e:
                            st.error(f"Kesalahan: {e}")
                        finally:
                            st.session_state.julia_process = None
                            st.session_state.stop_requested = False
                            status_text.text("Monitoring dihentikan.")
                            progress_bar.progress(0)
                    except:
                        st.error("Fail to save inversion parameters!")

                    # Tombol untuk menghentikan proses
                    if st.button("Hentikan Proses"):
                        if st.session_state.julia_process and st.session_state.julia_process.poll() is None:
                            st.session_state.stop_requested = True
                        else:
                            st.warning("Tidak ada proses yang sedang berjalan.")

                if "max_iteration_in_progress" not in st.session_state:
                    st.session_state.max_iteration_in_progress = None
                
                if "current_model_show" not in st.session_state:
                    st.session_state.model_current = None

                if "current_gradient_show" not in st.session_state:
                    st.session_state.current_gradient_show = None

                DIR_PROGRESS = "results/progress"
                file_model_current = os.listdir(DIR_PROGRESS)

                if len(file_model_current) >0:
                    current_view_model_selected = st.selectbox("Choose current model to show", file_model_current)

                    

                    model_current_path = os.path.join(DIR_PROGRESS, current_view_model_selected)
                    with h5py.File(model_current_path, 'r') as f:
                        # Akses dataset sebagai array
                        n = st.session_state.init_model_imaging["n"]
                        slowness_curremt = f["slowness"][:].reshape(n[::-1])
                        velocity_current = np.sqrt(1/slowness_curremt)
                        gradient_current = f["gradient"][:].reshape(n[::-1])
                    
                    st.divider()
                    #initial model show
                    # Heatmap
                    # Buat koordinat x dan z berdasarkan grid
                    x = st.session_state.init_model_imaging["o"][0] + np.arange(st.session_state.init_model_imaging["n"][0]) * st.session_state.init_model_imaging["d"][0] # array: [0, 10, 20, ..., 990]
                    z = st.session_state.init_model_imaging["o"][1] + np.arange(st.session_state.init_model_imaging["n"][1]) * st.session_state.init_model_imaging["d"][1]
                    col31, col32, col33 = st.columns([1,4,1])

                    with col32:
                        fig1 = plot_velocity_model(x, z, st.session_state.init_model_imaging["m"]*1e-3, title="Initial Model", min_value=None, max_value=None)
                        st.plotly_chart(fig1, key="init_model")
                        fig2 = plot_velocity_model(x, z, velocity_current, title=f"Model Iteration -> {current_view_model_selected}", min_value=None, max_value=None)
                        st.plotly_chart(fig2, key="current_velocity")
                        fig3 = plot_velocity_model(x, z, gradient_current, title=f"Gradient Iteration -> {current_view_model_selected}", min_value=None, max_value=None)
                        st.plotly_chart(fig3, key="current_gradient")



    else:
        st.error("Under Construction!")


# --- Halaman: Visualize ---
elif menu == "4. Visualize":
    st.title("📊 Visualize")
    st.write("Tampilkan hasil visualisasi model dan imaging.")

    vis_option = st.selectbox("Pilih data untuk divisualisasikan", ["Model Velocity", "Imaging Result", "Shot Position"])
    st.write(f"Menampilkan visualisasi untuk: {vis_option}")
    st.image("https://via.placeholder.com/400x200?text=Visualization+Output", caption="Visualisasi Data (Dummy)")
