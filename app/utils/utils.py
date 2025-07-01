import segyio
import numpy as np
from collections import defaultdict
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(gather_2d, dt, lowcut, highcut, order=4):
    """
    Terapkan filter bandpass Butterworth per kolom (tiap trace),
    input berbentuk (n_samples, n_traces).

    Parameters:
    - gather_2d: array (n_samples, n_traces)
    - dt: interval sampling (detik)
    - lowcut: frekuensi bawah (Hz)
    - highcut: frekuensi atas (Hz)
    - order: orde filter

    Returns:
    - filtered_gather: array (n_samples, n_traces)
    """
    nyq = 0.5 / dt
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Filter langsung per kolom (tiap trace)
    filtered = filtfilt(b, a, gather_2d, axis=0)
    return filtered



def extract_amplitude_and_spectrum(gather_2d, dt):
    """
    Ekstrak amplitude (rms) dan spektrum frekuensi dari 2D gather.
    
    Parameters:
    - gather_2d: numpy array dengan shape (n_traces, n_samples)
    - dt: sampling interval (dalam detik)
    
    Returns:
    - rms_amplitude: array (n_samples,), nilai RMS dari semua trace
    - freq: array frekuensi (Hz)
    - spectrum: array spektrum magnitude (rata-rata dari semua trace)
    """
    # RMS Amplitude per waktu
    rms_amplitude = np.sqrt(np.mean(gather_2d ** 2, axis=0))

    # FFT
    fft_vals = np.fft.rfft(gather_2d, axis=1)  # FFT per trace
    spectrum = np.mean(np.abs(fft_vals), axis=0)  # Mean magnitude
    freq = np.fft.rfftfreq(gather_2d.shape[1], dt)

    return rms_amplitude, freq, spectrum

def filter_all_shot_data(gathers, dt, lowcut, highcut, order=4):
    gathers_filter = {}
    for key in gathers.keys():
        # print(key)
        gathers_filter[key] = butter_bandpass_filter(gathers[key], dt, lowcut, highcut, order=4)
        
    return gathers_filter

def set_header_and_data(filepath, data, df, byte_location=None):

    #stack data
    data_stack = np.concatenate(list(data.values()), axis=1)
    
    with segyio.open(filepath, "r+", ignore_geometry=True) as f:
        for i in range(f.tracecount):
            hdr = f.header[i]
            coord_scalar = hdr.get(segyio.TraceField.SourceGroupScalar, 1)

            #prepare header
            trace_number_i = np.int32(df.iloc[i]["trace_number"])            
            ffid_i = np.int32(df.iloc[i]["ffid"])
            
            if coord_scalar < 0:
                coord_scalar = abs(coord_scalar)
           
                sx_i = np.int32(df.iloc[i]["sx"]*coord_scalar)   
                sy_i = np.int32(df.iloc[i]["sy"]*coord_scalar)   
                gx_i = np.int32(df.iloc[i]["gx"]*coord_scalar)   
                gy_i = np.int32(df.iloc[i]["gy"]*coord_scalar)   

            
            else:
                sx_i = np.int32(df.iloc[i]["sx"]*coord_scalar)   
                sy_i = np.int32(df.iloc[i]["sy"]*coord_scalar)   
                gx_i = np.int32(df.iloc[i]["gx"]*coord_scalar)   
                gy_i = np.int32(df.iloc[i]["gy"]*coord_scalar)

            #update header
            if byte_location is None:
                f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = trace_number_i
                f.header[i][segyio.TraceField.FieldRecord] = ffid_i
                f.header[i][segyio.TraceField.SourceX] = sx_i
                f.header[i][segyio.TraceField.SourceY] = sy_i
                f.header[i][segyio.TraceField.GroupX] = gx_i
                f.header[i][segyio.TraceField.GroupY] = gy_i

            else:
                f.header[i][byte_location["trace_number"]] = trace_number_i
                f.header[i][byte_location["ffid"]] = ffid_i
                f.header[i][byte_location["sx"]] = sx_i
                f.header[i][byte_location["sy"]] = sy_i
                f.header[i][byte_location["gx"]] = gx_i
                f.header[i][byte_location["gy"]] = gy_i

            #update data
            f.trace[i] = np.float32(data_stack[:, i])
            
def read_segy(filepath, byte_location=None):
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        f.mmap()  # Percepat akses header/trace dengan memory mapping
        
        trace_number = []
        ffid = []
        sx = []
        sy = []
        gx = []
        gy = []
        gathers = defaultdict(list)

        for i in range(f.tracecount):
            hdr = f.header[i]
            coord_scalar = hdr.get(segyio.TraceField.SourceGroupScalar, 1)
            if byte_location is None:
                trace_number.append(hdr[segyio.TraceField.TRACE_SEQUENCE_LINE])
                ffid_i = hdr[segyio.TraceField.FieldRecord]
                ffid.append(ffid_i)
                sx_i = hdr[segyio.TraceField.SourceX]
                sy_i = hdr[segyio.TraceField.SourceY]
                gx_i = hdr[segyio.TraceField.GroupX]
                gy_i = hdr[segyio.TraceField.GroupY]
                
            else:
                trace_number.append(f.header[i][byte_location["trace_number"]])
                ffid_i = hdr[byte_location["ffid"]]
                ffid.append(ffid_i)
                sx_i = hdr[byte_location["sx"]]
                sy_i = hdr[byte_location["sy"]]
                gx_i = hdr[byte_location["gx"]]
                gy_i = hdr[byte_location["gy"]]

            # Terapkan skala koordinat
            if coord_scalar > 1:
                sx_i *= coord_scalar
                sy_i *= coord_scalar
                gx_i *= coord_scalar
                gy_i *= coord_scalar
            elif coord_scalar < 0:
                sx_i /= abs(coord_scalar)
                sy_i /= abs(coord_scalar)
                gx_i /= abs(coord_scalar)
                gy_i /= abs(coord_scalar)

            sx.append(sx_i)
            sy.append(sy_i)
            gx.append(gx_i)
            gy.append(gy_i)

            trace = f.trace[i]
            gathers[ffid_i].append(trace)

        for fid in gathers:
            gathers[fid] = np.stack(gathers[fid]).T

        trace_header = {
            "trace_number": trace_number,
            "ffid": ffid,
            "sx": sx,
            "sy": sy,
            "gx": gx,
            "gy": gy
        }
        dt = f.header[1][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        return trace_header, gathers, dt

def get_geom_uniq(trace_header):
    sx,idx_uniq_src = np.unique(trace_header["sx"], return_index=True)
    sy = np.array(trace_header["sy"])[idx_uniq_src]
    ffid = np.array(trace_header["ffid"])[idx_uniq_src]

    gx,idx_uniq_rec = np.unique(trace_header["gx"], return_index=True)
    gy = np.array(trace_header["sy"])[idx_uniq_rec] 
    
    return ffid, sx, sy, gx, gy

import plotly.graph_objects as go

def plot_velocity_model(x, z, model_data, title="2D Velocity Model", min_value=None, max_value=None):
    """
    Membuat heatmap 2D velocity model dengan Plotly dengan aspect ratio equal.

    Parameters:
    - x (array-like): Koordinat horizontal (Distance X).
    - z (array-like): Koordinat vertikal (Depth).
    - model_data (2D array): Matriks data kecepatan [Z x X].
    - title (str): Judul plot.
    - min_value (float, optional): Nilai minimum untuk skala warna.
    - max_value (float, optional): Nilai maksimum untuk skala warna.

    Returns:
    - fig (plotly.graph_objects.Figure): Objek plot heatmap.
    """
    heatmap = go.Heatmap(
        z=model_data,
        x=x,
        y=z,
        colorscale='Jet',
        colorbar=dict(title='Velocity (m/s)'),
        zmin=min_value,
        zmax=max_value
    )

    fig = go.Figure(data=heatmap)

    fig.update_yaxes(
        autorange="reversed",
        title="Depth (m)",
        tickformat=".1f",
        scaleanchor="x",  # Make y-axis use same scale as x
        scaleratio=1
    )
    fig.update_xaxes(
        title="Distance X (m)",
        tickformat=".1f"
    )

    fig.update_layout(
        title=title,
        title_x=0.5,
        height=600
    )

    return fig

