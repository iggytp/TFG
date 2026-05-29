import pyvo
import pandas as pd
import numpy as np

# 1. Conexión al servidor TAP de la ESA
service = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")

# Definimos las columnas en nomenclatura de la ESA 
columns = """
    g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
    g.phot_g_mean_flux, g.phot_g_mean_flux_error,
    g.phot_bp_mean_flux, g.phot_bp_mean_flux_error,
    g.phot_rp_mean_flux, g.phot_rp_mean_flux_error,
    g.classprob_dsc_combmod_star,
    ap.logg_gspphot
"""

# Filtros de calidad de paralaje, magnitud límite y probabilidad >= 95% de ser estrella
quality_filters = """
    WHERE g.parallax_over_error > 20 
      AND g.phot_g_mean_mag < 18 
      AND g.classprob_dsc_combmod_star >= 0.95
"""

print("1. Descargando datos...")

# Consultas a Gaia DR3
query_dwarfs = f"SELECT TOP 5000 {columns}, 'Enana' as clase_real FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id {quality_filters} AND ap.logg_gspphot > 4.0"
query_giants = f"SELECT TOP 5000 {columns}, 'Gigante' as clase_real FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id {quality_filters} AND ap.logg_gspphot < 3.5"
query_excluded = f"SELECT count(*) as total FROM gaiadr3.gaia_source AS g JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id {quality_filters} AND ap.logg_gspphot BETWEEN 3.5 AND 4.0"

# Ejecutamos consultas
df_d = service.search(query_dwarfs).to_table().to_pandas()
df_g = service.search(query_giants).to_table().to_pandas()

print("- Calculando objetos excluidos (Lanzando tarea ASÍNCRONA en la ESA, puede tardar 1-2 minutos)...")
job = service.submit_job(query_excluded) # Crea el trabajo en el servidor
job.run()                                # Inicia la ejecución
job.wait()                               # Espera sin cortar la conexión
count_excl = job.fetch_result().to_table().to_pandas()['total'][0]
print(f"- ¡Completado! Objetos excluidos calculados.")

# Unimos y mezclamos las filas 
df_final = pd.concat([df_d, df_g], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# 2. FEATURE ENGINEERING (Errores Fotométricos y Benchmark)
# Fórmula astronómica para pasar de error de flujo a error de magnitud (e_Gmag)
df_final['e_Gmag'] = 1.0857 * (df_final['phot_g_mean_flux_error'] / df_final['phot_g_mean_flux'])
df_final['e_BPmag'] = 1.0857 * (df_final['phot_bp_mean_flux_error'] / df_final['phot_bp_mean_flux'])
df_final['e_RPmag'] = 1.0857 * (df_final['phot_rp_mean_flux_error'] / df_final['phot_rp_mean_flux'])

# Magnitud Absoluta (Solo para el gráfico Benchmark)
df_final['abs_mag_g'] = df_final['phot_g_mean_mag'] + 5 + 5 * np.log10(df_final['parallax'] / 1000)


print(f"Muestra total balanceada: {len(df_final)} estrellas (5k Enanas / 5k Gigantes).")
print(f"Objetos excluidos (log g entre 3.5 y 4.0): {count_excl} registros.")


# Guardamos la muestra
df_final.to_csv("dataset_v1.csv", index=False)
print("\n¡'dataset_v1.csv' guardado con éxito!")
