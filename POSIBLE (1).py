import os
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import segyio
import pyvista as pv
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

#MODEL CHECKPOINT 
#EARLY STOPING
#PREPROCESAMIENTO DE DATOS

# 1. CONFIGURACIÓN DE RUTAS Y DETECCIÓN 
# =================================================================
ruta_script = os.path.dirname(os.path.abspath(__file__))
PATH_RAIZ = ruta_script.split('proyecto')[0] + 'proyecto'
las_files, txt_files, sgy_files = [], [], []
print(f"--- Escaneando proyecto en: {PATH_RAIZ} ---")
for root, dirs, files in os.walk(PATH_RAIZ):
    for f in files:
        fp = os.path.join(root, f)
        if f.lower().endswith('.las'): las_files.append(fp)
        elif f.lower().endswith('.txt'): txt_files.append(fp)
        elif f.lower().endswith('.sgy'): sgy_files.append(fp)
print(f"Se encontraron los siguientes archivos: {len(las_files)} LAS, {len(txt_files)} TXT, {len(sgy_files)} SEGY")


# 2. PROCESAMIENTO DE CURVAS 
# =================================================================
all_data = []
MAPEO = {'GR': ['GR', 'Gamma Ray'], 'RHOB': ['RHOB', 'Density', 'DEN'], 
         'DT': ['DT', 'Sonic'], 'AI': ['P-Impedance', 'AI'], 'PHIE': ['PHIE', 'Porosity']}

for f_las in las_files:
    nombre_well = os.path.splitext(os.path.basename(f_las))[0]
    try:
        l = lasio.read(f_las)
        df_well = l.df().reset_index().replace([-999.25, -9999, -455.6956], np.nan)
        
        if df_well['RHOB'].mean() > 500:
                df_well['RHOB'] = df_well['RHOB'] / 1000.0
        # Normalizar nombres de columnas
        for std, posib in MAPEO.items():
            for p in posib:
                if p in df_well.columns:
                    df_well.rename(columns={p: std}, inplace=True)
                    break
        # FiltrO si NO existe RHOB para evitar cargas innecesarias
        if 'RHOB' not in df_well.columns: continue
        df_well = df_well.dropna(subset=['RHOB']).copy()
        df_well['WELL_NAME'] = nombre_well
        df_well['MARKER_ZONE'] = 'Desconocido'

        # Vincular CHECKSHOTS 
        f_cs = [f for f in txt_files if nombre_well.lower() in f.lower() and 'check' in f.lower()]
        if f_cs:
            cs = pd.read_csv(f_cs[0], header=None, sep=r'\s+', names=['D', 'T'])
            df_well['TIME_TWT'] = interp1d(cs['D'], cs['T'], fill_value="extrapolate")(df_well['DEPTH'])
        all_data.append(df_well)
    except Exception as e: print(f"Error en {nombre_well}: {e}")

df_total = pd.concat(all_data, ignore_index=True).fillna(0)

# =================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DE LA IA
# =================================================================
# K-Means para generar etiquetas de litología (Arena y "Arcilla")
km = KMeans(n_clusters=2, random_state=42, n_init='auto')
feats_ia = [c for c in ['GR', 'RHOB', 'DT', 'AI', 'PHIE'] if c in df_total.columns]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_total[feats_ia])
clusters = km.fit_predict(X_scaled)

# Calculamos el GR media de cada grupo
gr_idx = feats_ia.index('GR')
mean_gr_cluster_0 = X_scaled[clusters == 0, gr_idx].mean()
mean_gr_cluster_1 = X_scaled[clusters == 1, gr_idx].mean()
if mean_gr_cluster_0 > mean_gr_cluster_1:
    df_total['LITOLOGIA_IA'] = 1 - clusters
else:
    df_total['LITOLOGIA_IA'] = clusters

y = df_total['LITOLOGIA_IA'].values

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n--- Entrenando Red Neuronal ---")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=64, batch_size=32 ,verbose=1)

# Curvas de Aprendizaje
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Aprendizaje (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Error (Loss)')
plt.title('Curva de Error')
plt.show()

#Matriz de confusión
print("La matriz de confusión es:")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

mc = confusion_matrix(y_test, y_pred)
print(mc)
# =================================================================
# 4. PLOTEO DE REGISTROS (TRACKS)
# =================================================================
def graficar_pozos():
    for well in df_total['WELL_NAME'].unique()[:4]: 
        data = df_total[df_total['WELL_NAME'] == well].sort_values('DEPTH')
        fig, ax = plt.subplots(1, 4, figsize=(12, 10), sharey=True)
        ax[0].set_ylim(data['DEPTH'].max(), data['DEPTH'].min()) 
        ax[0].set_ylabel('Profundidad (ft/m)')

        #Track de GR
        ax[0].plot(data['GR'], data['DEPTH'], color='green', lw=1.5)
        ax[0].set_title('Gamma Ray (API)\n(0 - 150 API', fontsize=10, pad=15 )
        ax[0].set_xlim(0, 150)
        ax[0].xaxis.set_label_position('top')
        ax[0].xaxis.tick_top() 
        ax[0].grid(True, linestyle='--')
        
        #Track de densidad
        
        if 'RHOB' in data.columns:
            df_rhob = data[['RHOB', 'DEPTH']].dropna()
            df_rhob = df_rhob[(df_rhob['RHOB'] > 0.5) & (df_rhob['RHOB'] < 4.0)]
            
            if not df_rhob.empty:
                ax[1].plot(df_rhob['RHOB'], df_rhob['DEPTH'], color='red', lw=1.2)
                ax[1].set_title('RHOB\n(g/cc)', fontsize=10, pad=15)
            else:
                ax[1].text(0.5, 0.5, 'Sin datos\nválidos', transform=ax[1].transAxes, ha='center')
        else:
            ax[1].text(0.5, 0.5, 'RHOB no\nencontrado', transform=ax[1].transAxes, ha='center')

        ax[1].set_xlim(1.2, 2.95) # Escala estándar para g/cm3
        ax[1].xaxis.tick_top()
        ax[1].grid(True, linestyle='--')

        #Track sónico
        if 'AI' in data.columns and data['DT'].sum() != 0:
            ax[2].plot(data['DT'], data['DEPTH'], color='blue', lw=1.5)
        ax[2].set_title('DT\n(us/m)', fontsize=10, pad=15)
        ax[2].xaxis.tick_top()
        ax[2].grid(True, linestyle='--')


        # Litología IA
        lito_img = data['LITOLOGIA_IA'].values.reshape(-1, 1)
        cmap_lito = mcolors.ListedColormap(['yellow', 'gray'])
        
        lito_img = data['LITOLOGIA_IA'].values.reshape(-1, 1)
        ax[3].imshow(np.repeat(lito_img, 20, axis=1), aspect='auto', 
                      extent=[0, 1, data['DEPTH'].max(), data['DEPTH'].min()], 
                      cmap=cmap_lito)
        ax[3].set_title('Litología IA\n(Amarillo:Arena, Gris:Arcilla)', fontsize=10, pad=15)
        ax[3].set_xticks([]) 

        plt.tight_layout()
        plt.show()

graficar_pozos()

# =================================================================
# 5. INTEGRACIÓN FINAL CON CUBO SÍSMICO
# =================================================================
from scipy.ndimage import gaussian_filter

def visualizar_cubo_limpio(cubo_data, spacing=(20, 20, 2)):
    print("Limpiando ruido sismico...")
    cubo_suave = gaussian_filter(cubo_data, sigma=1)
    
    nx, ny, nz = cubo_suave.shape
    vmax = np.percentile(np.abs(cubo_suave), 98)
    
    # Imagen de cortes sismicos
    print("Generando imagen ")
    grid_stat = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1, 1, 1))
    grid_stat.point_data["amplitud"] = cubo_suave.flatten(order="F")

    p_stat = pv.Plotter(off_screen=True)
    # Cortes en el centro exacto
    p_stat.add_mesh(grid_stat.slice(normal='x', origin=(nx//2, 0, 0)), cmap="seismic", clim=[-vmax, vmax])
    p_stat.add_mesh(grid_stat.slice(normal='y', origin=(0, ny//2, 0)), cmap="seismic", clim=[-vmax, vmax])
    p_stat.add_mesh(grid_stat.slice(normal='z', origin=(0, 0, nz//2)), cmap="seismic", clim=[-vmax, vmax])
    p_stat.set_background("white")
    p_stat.show(screenshot="Reporte Final.png")
    plt.figure(figsize=(7, 7))
    plt.imshow(plt.imread("Reporte Final.png"))
    plt.axis('off')
    plt.show()

    print("Abriendo Explorador Interactiva...")
    grid_inter = pv.ImageData()
    grid_inter.dimensions = np.array(cubo_suave.shape)
    grid_inter.spacing = spacing
    grid_inter.point_data["Amplitud"] = cubo_suave.flatten(order="F")

    p = pv.Plotter(title="Cubo Sismico Integrado")
    p.set_background("white")
    p.add_mesh_slice_orthogonal(grid_inter, scalars="Amplitud", cmap="seismic", clim=[-vmax, vmax])
    p.add_mesh(grid_inter.outline(), color="black", line_width=2)
    # Pozos centrados en el volumen
    z_max = grid_inter.dimensions[2] * grid_inter.spacing[2]
    centro_x = (grid_inter.dimensions[0] * grid_inter.spacing[0]) / 2
    centro_y = (grid_inter.dimensions[1] * grid_inter.spacing[1]) / 2

    for i, name in enumerate(df_total['WELL_NAME'].unique()[:4]): # Max 4 pozos para no saturar
        offset = i * 150
        pozo = pv.Line([centro_x + offset, centro_y + offset, 0], 
                       [centro_x + offset, centro_y + offset, z_max])
        p.add_mesh(pozo, color='black', line_width=3, render_lines_as_tubes=True)
        p.add_point_labels([centro_x + offset, centro_y + offset, 0], [name], font_size=10)
    p.add_axes() 
    
    p.camera.up = (0, 0, -1) 
    p.camera_position = 'iso'
    p.show()
if len(sgy_files) > 0:
    with segyio.open(sgy_files[0], iline=189, xline=193, ignore_geometry=True) as s:
        il_u = np.unique(s.attributes(segyio.TraceField.INLINE_3D)[:])
        xl_u = np.unique(s.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        cubo_final = np.zeros((len(il_u), len(xl_u), len(s.samples)))
        il_m = {v: i for i, v in enumerate(il_u)}
        xl_m = {v: i for i, v in enumerate(xl_u)}
        for i, (l_i, l_x) in enumerate(zip(s.attributes(189)[:], s.attributes(193)[:])):
            cubo_final[il_m[l_i], xl_m[l_x], :] = s.trace[i]
    
    visualizar_cubo_limpio(cubo_final)